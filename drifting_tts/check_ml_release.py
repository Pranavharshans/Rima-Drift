"""Check selected Malayalam v3.2 components and write a provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .models.prosody_net import tts_fingerprint
from .text import symbols_for
from .train import load_tts

SPEAKERS = {"malayalam_female": 0, "malayalam_male": 1}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--acoustic", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--implementation-commit", required=True)
    p.add_argument("--heldout-female", help="prosody val results.json for the female voice")
    p.add_argument("--heldout-male", help="prosody val results.json for the male voice")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(acoustic: Path, vocoder: Path, prosody: Path) -> dict:
    acoustic = acoustic.resolve(strict=True)
    vocoder = vocoder.resolve(strict=True)
    prosody = prosody.resolve(strict=True)
    model, cfg, stats = load_tts(acoustic, "cpu")
    if cfg.model.get("language") != "ml" or stats.get("language") != "ml":
        raise ValueError("acoustic checkpoint and stats must both be Malayalam")
    if stats.get("backend") != "bigvgan" or not model.pitch_enabled:
        raise ValueError("Malayalam v3.2 requires bigvgan mels and pitch conditioning")
    acoustic_ck = torch.load(acoustic, map_location="cpu", weights_only=False)
    if acoustic_ck.get("symbols") != symbols_for("ml"):
        raise ValueError("acoustic checkpoint has the wrong Malayalam vocabulary")
    speaker_file = Path(cfg.data.root) / "speakers.json"
    speaker_map = json.loads(speaker_file.read_text())
    if speaker_map != SPEAKERS or acoustic_ck.get("num_speakers") != len(SPEAKERS):
        raise ValueError(f"expected two mapped Malayalam voices, got {speaker_map}")

    vc = torch.load(vocoder, map_location="cpu", weights_only=False)
    if not isinstance(vc.get("vocos"), dict) or vc.get("mel") != "bigvgan":
        raise ValueError("vocoder must be a Vocos export for bigvgan mels")
    if vc.get("head_padding") != "same" or not isinstance(vc.get("step"), int):
        raise ValueError("vocoder has unexpected framing or missing step metadata")

    pc = torch.load(prosody, map_location="cpu", weights_only=False)
    if not {"ema", "stats", "net_cfg", "cond_dim", "tts_fingerprint"} <= pc.keys():
        raise ValueError("prosody export lacks inference state or acoustic fingerprint")
    fingerprint = tts_fingerprint(model)
    if pc["tts_fingerprint"] != fingerprint:
        raise ValueError("prosody was trained against another acoustic encoder")
    if Path(pc.get("tts", "")).resolve() != acoustic:
        raise ValueError("prosody provenance names another acoustic checkpoint")
    if set(map(int, pc.get("duration_scales", {}))) != set(SPEAKERS.values()):
        raise ValueError("prosody lacks calibration for both Malayalam voices")
    if pc.get("net_cfg", {}).get("kind") != "drift":
        raise ValueError("prosody checkpoint is not the v3.2 drift predictor")

    files = {"acoustic": acoustic, "vocoder": vocoder, "prosody": prosody}
    return {"language": "ml", "sample_rate": 24000, "mel_backend": "bigvgan", "speakers": SPEAKERS,
            "acoustic_fingerprint": fingerprint,
            "checkpoints": {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                            for name, path in files.items()},
            "steps": {"acoustic": acoustic_ck.get("step"), "vocoder_continuation": vc["step"],
                      "prosody": pc.get("step")}}


def validate_heldout(path: Path, speaker: int, acoustic: Path, vocoder: Path, prosody: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("split") != "val" or report.get("speaker") != speaker:
        raise ValueError(f"{path}: expected Rasa val results for speaker {speaker}")
    if report.get("num_utterances", 0) < 1:
        raise ValueError(f"{path}: no held-out utterances")
    if Path(report.get("model", "")).resolve() != acoustic.resolve():
        raise ValueError(f"{path}: acoustic checkpoint differs from release")
    if Path(report.get("vocoder", "")).resolve() != vocoder.resolve():
        raise ValueError(f"{path}: vocoder checkpoint differs from release")
    if Path(report.get("release", {}).get("prosody", "")).resolve() != prosody.resolve():
        raise ValueError(f"{path}: prosody checkpoint differs from release")
    rows = {r.get("system"): r for r in report.get("rows", [])}
    if not {"recording", "copy", "predicted", "release"} <= rows.keys():
        raise ValueError(f"{path}: missing recording, copy, deterministic or sampled row")
    for name in ("predicted", "release"):
        for metric in ("wer", "cer"):
            value = rows[name].get(metric)
            if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                raise ValueError(f"{path}: {name} has no valid Malayalam {metric}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "utterances": report["num_utterances"],
            "deterministic_wer": rows["predicted"]["wer"], "sampled_wer": rows["release"]["wer"],
            "deterministic_cer": rows["predicted"]["cer"], "sampled_cer": rows["release"]["cer"]}


def run(args: argparse.Namespace) -> None:
    report = validate(Path(args.acoustic), Path(args.vocoder), Path(args.prosody))
    report["dataset_revision"] = args.dataset_revision
    report["implementation_commit"] = args.implementation_commit
    if bool(args.heldout_female) != bool(args.heldout_male):
        raise ValueError("supply both female and male held-out reports")
    if args.heldout_female:
        report["heldout"] = {
            "malayalam_female": validate_heldout(Path(args.heldout_female), 0, Path(args.acoustic),
                                                   Path(args.vocoder), Path(args.prosody)),
            "malayalam_male": validate_heldout(Path(args.heldout_male), 1, Path(args.acoustic),
                                                 Path(args.vocoder), Path(args.prosody)),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Malayalam release components match: {out}")
