"""Train the mel bandwidth extension (:class:`drifting_tts.bwe.MelBWE`) on full-band 24 kHz audio.

Reads ``audio.bin`` of prepared datasets (``prepare --save-audio``), only training utterances whose measured bandwidth
(``scores.jsonl`` ``bw_hz``, ``drifting-tts score``) reaches ``--min-bandwidth``. Every clip is low-passed the way a
16 kHz source is: 24 -> 16 kHz with a random roll-off (the source's anti-aliasing filter), then back to 24 kHz as
``prepare`` resamples. The net learns the full log-mel from the band-limited one (L1, the band above 6 kHz weighted
up); at inference only the bins above the cutoff are taken from it.

    drifting-tts train-bwe --data data/tr12_eleven --workdir runs/bwe
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .audio import SAMPLE_RATE, BigVGANLogMel
from .bwe import MelBWE
from .data import load_scores


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", nargs="+", required=True, help="prepared datasets with audio.bin and scores.jsonl")
    p.add_argument("--workdir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seconds", type=float, default=2.0, help="clip length")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--min-bandwidth", type=float, default=11000.0, help="bw_hz of the clips used (full band only)")
    p.add_argument("--rolloff", type=float, nargs=2, default=[0.85, 0.99],
                   help="range of the 16 kHz anti-aliasing roll-off (fraction of 8 kHz)")
    p.add_argument("--cutoff-hz", type=float, default=7600.0, help="default cutoff stored for inference")
    p.add_argument("--val", type=int, default=64, help="held-out clips (val split) for the report")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6, help="clip loading processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


class Clips(torch.utils.data.IterableDataset):
    """Random fixed-length 24 kHz clips of the full-band utterances of some prepared datasets (endless)."""

    def __init__(self, roots: list[str], split: str, min_bw: float, samples: int):
        self.items, self.audio = [], []
        for k, root in enumerate(map(Path, roots)):
            scores = load_scores(root)
            self.audio.append(np.memmap(root / "audio.bin", dtype=np.float16, mode="r"))
            with open(root / "index.jsonl") as f:
                for i, line in enumerate(f):
                    e = json.loads(line)
                    if (e["split"] == split and e.get("audio_samples", 0) >= samples
                            and scores.get(i, {}).get("bw_hz", 0) >= min_bw):
                        self.items.append((k, e["audio_offset"], e["audio_samples"]))
        self.items = sorted(set(self.items))  # merged roots list repeated utterances more than once
        self.samples = samples

    def sample(self, n: int, rng: random.Random) -> torch.Tensor:
        out = np.empty((n, self.samples), dtype=np.float32)
        for j in range(n):
            k, off, length = self.items[rng.randrange(len(self.items))]
            start = off + rng.randrange(length - self.samples + 1)
            out[j] = self.audio[k][start: start + self.samples]
        return torch.from_numpy(out)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(torch.initial_seed() + (info.id if info else 0))
        while True:
            yield self.sample(1, rng)[0]


def degrade(wav: torch.Tensor, rolloff: float) -> torch.Tensor:
    """A 24 kHz clip as a 16 kHz source (anti-aliasing ``rolloff``) resampled back to 24 kHz."""
    import torchaudio

    low = torchaudio.functional.resample(wav, SAMPLE_RATE, 16000, rolloff=rolloff)
    return torchaudio.functional.resample(low, 16000, SAMPLE_RATE)[..., : wav.shape[-1]]


def run(args) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    samples = int(args.seconds * SAMPLE_RATE)
    train, val = (Clips(args.data, s, args.min_bandwidth, samples) for s in ("train", "val"))
    print(f"train-bwe: {len(train.items)} training utterances, {len(val.items)} held out (bw >= "
          f"{args.min_bandwidth:.0f} Hz)", flush=True)
    if not val.items:  # no held-out split: report on training clips
        val = train
    logmel = BigVGANLogMel().to(dev)
    model = MelBWE(dim=args.dim).to(dev)
    ema = MelBWE(dim=args.dim).to(dev).eval()
    ema.load_state_dict(model.state_dict())
    print(f"MelBWE: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    weight = (1 + 4 * (model.centres > 6000).float())[None, :, None].to(dev)
    vrng = random.Random(1)
    vwav = val.sample(args.val, vrng).to(dev)
    vin, vtg = logmel(degrade(vwav, 0.95)), logmel(vwav)
    high = (model.centres > 8000).to(dev)

    def report(net) -> dict:
        with torch.no_grad():
            out = net.extend(vin, args.cutoff_hz)
        return {"high_l1_in": float((vin - vtg)[:, high].abs().mean()),
                "high_l1_out": float((out - vtg)[:, high].abs().mean()),
                "low_l1_out": float((out - vtg)[:, ~high].abs().mean())}

    batches = iter(torch.utils.data.DataLoader(train, batch_size=args.batch, num_workers=args.workers,
                                              pin_memory=True, persistent_workers=args.workers > 0))
    t0, run_loss = time.time(), 0.0
    for step in range(1, args.steps + 1):
        wav = next(batches).to(dev, non_blocking=True)
        with torch.no_grad():
            x, y = logmel(degrade(wav, rng.uniform(*args.rolloff))), logmel(wav)
        loss = ((model(x) - y).abs() * weight).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        run_loss += loss.item()
        if step % args.log_every == 0:
            print(f"step {step} loss={run_loss / args.log_every:.4f} " + " ".join(
                f"{k}={v:.4f}" for k, v in report(ema).items()) + f" {step / (time.time() - t0):.1f} it/s",
                flush=True)
            run_loss = 0.0
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": ema.state_dict(), "config": ema.config, "cutoff_hz": args.cutoff_hz, "step": step,
                        "report": report(ema)}, work / "bwe.pt")
    print(f"wrote {work / 'bwe.pt'}: {report(ema)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_args(p)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
