"""Inspect a trained 2-D Mel-MAE checkpoint on remote held-out Malayalam mels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import MelDataset
from .train_mae import load_mae


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/mae2d_ml.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("dev", "val"), default="dev")
    p.add_argument("--per-speaker", type=int, default=4)
    p.add_argument("--expected-step", type=int, default=None)
    p.add_argument("--out", required=True, help="remote-only report directory; includes gated mel samples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ds = MelDataset(cfg.data.root, args.split, min_quality=0, min_frames=cfg.train.crop_frames,
                    max_frames=cfg.data.max_frames)
    if cfg.data.get("language", "tr") != ds.language or cfg.model.n_mels != ds.dim:
        raise ValueError("Mel-MAE config does not match the prepared dataset")
    model = load_mae(args.checkpoint, args.device, expected_backend=ds.backend,
                     expected_language=ds.language, expected_mels=ds.dim)
    if not hasattr(model, "reconstruct"):
        raise ValueError("check-mae requires a 2-D Mel-MAE with reconstruct()")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.expected_step is not None and ck.get("step") != args.expected_step:
        raise ValueError(f"checkpoint step {ck.get('step')} != {args.expected_step}")
    rows: dict[str, list[dict]] = defaultdict(list)
    examples = {}
    for i, entry in enumerate(ds.items):
        speaker = entry["speaker"]
        if len(rows[speaker]) >= args.per_speaker:
            continue
        mel = ds[i]["mel"][:, : cfg.train.crop_frames][None].to(args.device)
        torch.manual_seed(cfg.train.seed + i)
        recon, mask, _ = model.reconstruct(mel, cfg.train.mask_ratio)
        if recon.shape != mel.shape or mask.shape != mel.shape:
            raise ValueError(f"reconstruction shape mismatch for {speaker}")
        if not torch.isfinite(recon).all():
            raise ValueError(f"non-finite reconstruction for {speaker}")
        hidden = mask.sum().clamp_min(1)
        mse = float((((recon - mel) ** 2) * mask).sum() / hidden)
        zero_baseline = float(((mel**2 * mask).sum()) / hidden)
        if not math.isfinite(mse):
            raise ValueError(f"non-finite masked MSE for {speaker}")
        rows[speaker].append({"index": i, "masked_mse": mse, "zero_baseline_mse": zero_baseline})
        if speaker not in examples:
            examples[speaker] = {"mel": mel[0].cpu().numpy(), "recon": recon[0].cpu().numpy(),
                                 "mask": mask[0].cpu().numpy()}
        if all(len(v) >= args.per_speaker for v in rows.values()) and len(rows) == ds.num_speakers:
            break
    if len(rows) != ds.num_speakers or any(len(v) < args.per_speaker for v in rows.values()):
        raise ValueError(f"too few {args.split} examples per speaker: { {k: len(v) for k, v in rows.items()} }")
    acts = model.get_activations(mel, every_k_block=2, spectral_detail=True)
    if not acts or not all(torch.isfinite(v).all() for v in acts.values()):
        raise ValueError("non-finite or missing Mel-MAE activations for the acoustic loss")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reconstruction_samples.npz",
                        **{f"{speaker}_{kind}": value for speaker, parts in examples.items()
                           for kind, value in parts.items()})
    report = {"checkpoint": str(args.checkpoint), "step": ck.get("step"), "split": args.split,
              "language": ds.language, "backend": ds.backend, "n_mels": ds.dim,
              "activation_maps": len(acts),
              "per_speaker": {k: {"samples": v, "mean_masked_mse": sum(x["masked_mse"] for x in v) / len(v),
                                  "mean_zero_baseline_mse": sum(x["zero_baseline_mse"] for x in v) / len(v)}
                              for k, v in rows.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
