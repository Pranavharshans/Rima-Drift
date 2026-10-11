"""Export the model to ONNX for the browser (onnxruntime-web, WebGPU) and check every graph against PyTorch.

Three graphs; the glue between them is a few lines of JavaScript (web/tts.js):

* ``text_encoder.onnx``: token ids ``[1, N]``, speaker ``[1]`` -> token condition ``[1, 100 + d, N]`` (prior mean and
  pitch-conditioned text features) and log-durations ``[1, 1, N]``;
* ``generator.onnx``: noise ``[1, 100, T]``, frame condition ``[1, 100 + d, T]``, speaker, guidance scale ``[1]``
  and style codes ``[1, noise_coords]`` -> log-mel ``[1, 100, T]`` (one DriftDiT pass, de-normalised);
* ``vocoder.onnx``: log-mel ``[1, 100, T]`` -> waveform ``[1, T * 256]`` at 24 kHz: a BigVGAN (weight norm removed)
  or a Vocos on BigVGAN mels (``vocos-v2``, 13.5 M), its ISTFT head in plain operators (:class:`VocosVocoder`).

Between the encoder and the generator, each token's condition column is repeated ``ceil(exp(logw) * scale)`` times
(the hard alignment of ``durations_to_alignment``). The graphs are simplified with onnxslim when it is installed.
``--fp16`` also writes ``*_fp16.onnx`` copies that store the weights in half precision (half the download) and cast
them back to fp32 when the session is created, so they compute like the fp32 graphs on any WebGPU device. The web demo
loads the fp32 text encoder (15 MB more; it carries most of the rounding error) and the fp16 generator and vocoder:
``config.json["web"]`` names the files.

    python scripts/export_onnx.py --model drifting_tts_v3.1.pt --vocoder bigvgan_v2_ft.pt --out web/models
    python scripts/export_onnx.py --only vocoder --vocoder vocos-v2 --out onnx/vocos_v2   # the vocoder graph alone
"""

import argparse
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from drifting_tts.audio import HOP_LENGTH, N_FFT
from drifting_tts.text import SYMBOLS
from drifting_tts.train import load_tts
from drifting_tts.vocoder import load_vocoder
from drifting_tts.voices import DEFAULT_VOICE, VOICES


@contextlib.contextmanager
def unmasked_attention():
    """One utterance has no padding, so the encoder's attention mask is all ones. Dropping it keeps the exported graph
    free of the IsNaN / Where guard for fully masked rows, which onnxruntime-web runs on the CPU: a GPU round trip in
    every layer (30 ms -> a few ms for the text encoder in the browser)."""
    sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, **kw: sdpa(q, k, v, **kw)
    try:
        yield
    finally:
        F.scaled_dot_product_attention = sdpa


class TextEncoder(nn.Module):
    def __init__(self, tts):
        super().__init__()
        self.tts = tts

    def forward(self, text: torch.Tensor, speaker: torch.Tensor):
        lengths = torch.ones_like(speaker) * text.shape[1]
        h, mu, logw, x_mask = self.tts.encoder(text, lengths, speaker)
        if self.tts.pitch_enabled:
            h, _ = self.tts.pitch_condition(h, x_mask, speaker)
        return torch.cat([mu, h], 1), logw


class Generator(nn.Module):
    def __init__(self, tts, mean: float, std: float):
        super().__init__()
        self.tts, self.mean, self.std = tts, mean, std

    def forward(self, z, cond, speaker, cfg_scale, noise_labels):
        mel = self.tts.generate(z, cond, speaker, cfg_scale, noise_labels=noise_labels)
        return mel * self.std + self.mean


class Vocoder(nn.Module):
    def __init__(self, bigvgan):
        super().__init__()
        self.bigvgan = bigvgan
        static_channel_filters(bigvgan)

    def forward(self, mel):
        return self.bigvgan(mel)[:, 0].clamp(-1, 1)


class VocosVocoder(nn.Module):
    """A Vocos on BigVGAN mels (``Vocoder._vocos_bigvgan``) with its ISTFT head in plain operators, so the graph needs
    no DFT / STFT operator: the inverse real FFT of every frame is a product with a windowed DFT basis, and the
    overlap-add, ``n_fft = 4 hop``, is four shifted adds; the squared-window envelope is built the same way."""

    def __init__(self, vocos: nn.Module):
        super().__init__()
        self.backbone, self.out = vocos.backbone, vocos.head.out
        self.centred = vocos.head.istft.padding == "center"
        n, hop = N_FFT, HOP_LENGTH
        assert n % hop == 0, "the overlap-add assumes n_fft is a multiple of the hop"
        window = vocos.head.istft.window.double()
        k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
        ang = 2 * math.pi * k * torch.arange(n, dtype=torch.float64)[None] / n
        scale = torch.full((n // 2 + 1, 1), 2.0 / n, dtype=torch.float64)
        scale[0] = scale[-1] = 1.0 / n  # irfft: the DC and Nyquist bins count once (their imaginary parts not at all)
        self.register_buffer("cos_basis", (torch.cos(ang) * scale * window).float())  # [n // 2 + 1, n]
        self.register_buffer("sin_basis", (-torch.sin(ang) * scale * window).float())
        self.register_buffer("window_sq", window.square().float())
        self.pad = n // 2 if self.centred else (n - hop) // 2

    @staticmethod
    def overlap_add(frames: torch.Tensor) -> torch.Tensor:
        """``[B, T, n_fft]`` frames at hop ``HOP_LENGTH`` -> ``[B, (T - 1) * hop + n_fft]`` samples."""
        B, T, n = frames.shape
        r = n // HOP_LENGTH
        chunks = frames.reshape(B, T, r, HOP_LENGTH)
        y = sum(F.pad(chunks[:, :, j], (0, 0, j, r - 1 - j)) for j in range(r))  # [B, T + r - 1, hop]
        return y.reshape(B, -1)

    def forward(self, mel):
        n = mel.shape[-1] * HOP_LENGTH
        x = F.pad(mel, (0, 1), mode="replicate") if self.centred else mel
        mag, p = self.out(self.backbone(x)).chunk(2, dim=-1)  # [B, T, n_fft // 2 + 1] each
        mag = torch.exp(mag).clamp(max=1e2)
        frames = (mag * torch.cos(p)) @ self.cos_basis + (mag * torch.sin(p)) @ self.sin_basis  # [B, T, n_fft]
        env = self.overlap_add(self.window_sq.expand(1, frames.shape[1], -1))
        y = self.overlap_add(frames) / env.clamp_min(1e-11)
        return y[:, self.pad: self.pad + n].clamp(-1, 1)


def static_channel_filters(bigvgan: nn.Module) -> None:
    """BigVGAN's anti-aliased resampling expands its filter to ``x.shape[1]`` channels, which the ONNX tracer keeps
    symbolic ("convolution for kernel of unknown shape"). The channel count is fixed, so freeze it at trace time."""
    import torch.nn.functional as F

    def up(self, x):
        c = int(x.shape[1])
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)
        return x[..., self.pad_left: -self.pad_right]

    def lowpass(self, x):
        c = int(x.shape[1])
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)

    for m in bigvgan.modules():
        name = type(m).__name__
        if name == "UpSample1d":
            m.forward = up.__get__(m)
        elif name == "LowPassFilter1d":
            m.forward = lowpass.__get__(m)


def export(module: nn.Module, args: tuple, path: Path, names: list[str], outputs: list[str], dynamic: dict) -> None:
    torch.onnx.export(module, args, str(path), input_names=names, output_names=outputs, dynamic_axes=dynamic,
                      opset_version=18, dynamo=False)


def simplify(path: Path) -> None:
    try:
        import onnxslim
    except ImportError:
        return
    import onnx

    onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))


def half_weights(src: Path, dst: Path, min_size: int = 1024) -> None:
    """Store large fp32 initializers as fp16, each followed by a Cast back to fp32 (folded at session creation).

    Depthwise ``[C, 1, K]`` kernels stay fp32: they are BigVGAN's anti-aliasing filters (2 MB), and rounding them
    costs more than all the other weights together."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    keep, casts = [], []
    for init in m.graph.initializer:
        depthwise = len(init.dims) == 3 and init.dims[1] == 1
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < min_size or depthwise:
            keep.append(init)
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def check(path: Path, module, feeds: dict, name: str) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    with torch.no_grad():
        ref = module(*feeds.values())
    ref = ref if isinstance(ref, tuple) else (ref,)
    err = max(float(np.abs(g - r.numpy()).max() / (np.abs(r.numpy()).max() + 1e-9)) for g, r in zip(got, ref))
    print(f"{name}: max relative error vs PyTorch {err:.2e}", flush=True)
    return err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", help="TTS checkpoint (not needed with --only vocoder)")
    p.add_argument("--vocoder", required=True,
                   help="registry name or checkpoint: a BigVGAN (bigvgan_v2_ft.pt, bigvgan-base-ft) or a Vocos on "
                        "BigVGAN mels (vocos-v2, vocos-v2-balanced, vocos-ft)")
    p.add_argument("--only", choices=["vocoder"], help="export and check the vocoder graph alone")
    p.add_argument("--out", default="web/models")
    p.add_argument("--fp16", action="store_true", help="also write *_fp16.onnx copies with half-precision weights")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    voc = load_vocoder(args.vocoder, "cpu")
    if voc.kind == "bigvgan":
        vocoder = Vocoder(voc.model).eval()
    elif voc.kind == "vocos" and voc.mel == "bigvgan":
        vocoder = VocosVocoder(voc.model).eval()
    else:
        raise SystemExit(f"{args.vocoder}: only BigVGAN and Vocos-on-BigVGAN-mel vocoders export to ONNX")
    mel = torch.randn(1, 100, 61) * 2 - 5
    export(vocoder, (mel,), out / "vocoder.onnx", ["mel"], ["audio"], {"mel": {2: "T"}, "audio": {1: "S"}})
    simplify(out / "vocoder.onnx")
    for t in (97, 32):  # against the PyTorch vocoder of the synthesizer (its torch.fft ISTFT for a Vocos)
        check(out / "vocoder.onnx", voc, {"mel": torch.randn(1, 100, t) * 2 - 5}, f"vocoder T={t}")
    voc_meta = {"vocoder": {"name": args.vocoder, "kind": voc.kind, "context": voc.context,
                            "parameters": voc.num_params}}
    if args.only == "vocoder":
        if args.fp16:
            half_weights(out / "vocoder.onnx", out / "vocoder_fp16.onnx")
        (out / "config.json").write_text(json.dumps({"sample_rate": 24000, "hop_length": 256, "n_mels": 100,
                                                     **voc_meta}, indent=1))
        print(f"wrote {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB"
                                           for f in sorted(out.glob("*.onnx"))))
        return
    if not args.model:
        raise SystemExit("--model is required unless --only vocoder")

    tts, cfg, stats = load_tts(args.model, "cpu")
    gen = tts.generator
    enc, generator = TextEncoder(tts).eval(), Generator(tts, stats["mean"], stats["std"]).eval()

    n, t = 23, 61  # odd lengths: the generator pads T to its patch size
    text = torch.randint(2, len(SYMBOLS), (1, n))
    spk = torch.tensor([VOICES[DEFAULT_VOICE]["id"]])
    cond, logw = enc(text, spk)
    gen_args = (torch.randn(1, 100, t), torch.randn(1, cond.shape[1], t), spk, torch.tensor([2.0]),
                torch.randint(0, gen.noise_classes, (1, max(1, gen.noise_coords))))

    with unmasked_attention():
        export(enc, (text, spk), out / "text_encoder.onnx", ["text", "speaker"], ["cond", "logw"],
               {"text": {1: "N"}, "cond": {2: "N"}, "logw": {2: "N"}})
    export(generator, gen_args, out / "generator.onnx", ["z", "cond", "speaker", "cfg_scale", "noise_labels"], ["mel"],
           {"z": {2: "T"}, "cond": {2: "T"}, "mel": {2: "T"}})

    for name in ("text_encoder", "generator"):
        simplify(out / f"{name}.onnx")

    # parity at other lengths than the export example (dynamic axes really dynamic)
    text2 = torch.randint(2, len(SYMBOLS), (1, 57))
    check(out / "text_encoder.onnx", enc, {"text": text2, "speaker": spk}, "text_encoder")
    cond2, _ = enc(text2, spk)
    for t2 in (150, 151):  # even and odd: with and without patch padding
        feeds = {"z": torch.randn(1, 100, t2), "cond": torch.randn(1, cond2.shape[1], t2), "speaker": spk,
                 "cfg_scale": torch.tensor([2.0]), "noise_labels": gen_args[4]}
        check(out / "generator.onnx", generator, feeds, f"generator T={t2}")

    if args.fp16:
        for name in ("text_encoder", "generator", "vocoder"):
            half_weights(out / f"{name}.onnx", out / f"{name}_fp16.onnx")

    half = "_fp16" if args.fp16 else ""
    meta = {"sample_rate": 24000, "hop_length": 256, "n_mels": 100, "cond_channels": int(cond.shape[1]),
            "noise_classes": gen.noise_classes, "noise_coords": max(1, gen.noise_coords),
            "duration_scale": tts.duration_scale,
            "duration_scales": {str(k): v for k, v in tts.duration_scales.items()},
            "temperature": float(getattr(tts, "temperature", None) or 0.3),
            "voices": {k: v["id"] for k, v in VOICES.items()}, "default_voice": DEFAULT_VOICE, **voc_meta,
            "files": {f.name: f.stat().st_size for f in sorted(out.glob("*.onnx"))},
            "web": {"text_encoder": "text_encoder.onnx", "generator": f"generator{half}.onnx",
                    "vocoder": f"vocoder{half}.onnx"}}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(out.glob("*.onnx"))))

if __name__ == "__main__":
    main()
