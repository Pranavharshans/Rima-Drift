"""Frame-level detail of generated VAE latents against the recordings (#27): over-smooth latents decode as noise.

For the first ``--num`` held-out utterances of one speaker in the latent data root, the model generates the latents
of the same texts (predicted durations with the voice's duration factor, noise temperature ``--temperature``, CFG
``--cfg``, seed = index). Generated and recorded latents are averaged back to native VAE frames (the ``latent_repeat``
copies, as ``LatentVocoder`` does) in normalised units, and compared on:

* ``change``: the mean absolute frame-to-frame change ``|x_t - x_{t-1}|`` over frames and channels (#27: recordings
  0.98, the 10k-step VoxCPM2 pilot 0.62, measured on other sentences);
* ``std``: the per-channel standard deviation over the utterance, averaged over channels (0.97 vs 0.68);
* ``hf``: the share of the frame-to-frame change spectrum above a quarter of the frame rate (fast detail).

    python scripts/latent_smoothness.py --model runs/tts_voxcpm2/model_ema.pt --num 100 --out smooth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drifting_tts.data import MelDataset
from drifting_tts.train import load_tts


def native(x: torch.Tensor, repeat: int) -> torch.Tensor:
    """``[C, T]`` repeated frames -> ``[C, T // repeat]`` native frames (mean of the copies)."""
    n = x.shape[-1] // repeat
    return x[:, : n * repeat].reshape(x.shape[0], n, repeat).mean(-1)


def stats(x: torch.Tensor) -> dict:
    d = x[:, 1:] - x[:, :-1]
    spec = torch.fft.rfft(d, dim=-1).abs().square().sum(0)
    q = spec.shape[0] // 2
    return {"change": float(d.abs().mean()), "std": float(x.std(-1).mean()),
            "hf": float(spec[q:].sum() / spec.sum().clamp_min(1e-12))}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="latent TTS checkpoint (model_ema.pt)")
    p.add_argument("--data", default=None, help="latent data root (default: the model's data.root)")
    p.add_argument("--split", default="val")
    p.add_argument("--speaker", type=int, default=722)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--out", default=None, help="JSON summary")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, cfg, _ = load_tts(args.model, args.device)
    ds = MelDataset(args.data or cfg.data.root, args.split, min_frames=1, max_frames=10**9)
    r = ds.latent_repeat
    items = [i for i in range(len(ds)) if ds.items[i]["spk_id"] == args.speaker][: args.num]
    if not items:
        raise SystemExit(f"no {args.split} utterances of speaker {args.speaker}")
    scale = model.duration_scales.get(args.speaker, model.duration_scale)
    rows = []
    for k, i in enumerate(items):
        it = ds[i]
        text = it["text"][None].to(args.device)
        spk = torch.tensor([args.speaker], device=args.device)
        g = torch.Generator(device=args.device).manual_seed(k)
        x, n = model.synthesize(text, torch.tensor([text.shape[1]], device=args.device), spk, cfg_scale=args.cfg,
                                temperature=args.temperature, length_scale=scale, generator=g)
        gen = stats(native(x[0, :, : int(n[0])].float().cpu(), r))
        rec = stats(native(it["mel"], r))
        rows.append({"gen": gen, "rec": rec})
    out = {"model": str(args.model), "n": len(rows), "speaker": args.speaker, "temperature": args.temperature,
           "cfg": args.cfg}
    for side in ("gen", "rec"):
        for key in ("change", "std", "hf"):
            out[f"{side}_{key}"] = float(np.mean([row[side][key] for row in rows]))
    for key in ("change", "std", "hf"):
        out[f"ratio_{key}"] = float(np.median([row["gen"][key] / max(row["rec"][key], 1e-9) for row in rows]))
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
