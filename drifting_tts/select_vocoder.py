"""Rank Malayalam Vocos snapshots using balanced development audio reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEAKERS = ("malayalam_female", "malayalam_male")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", required=True)


def rank_reports(paths: list[str]) -> list[dict]:
    ranked = []
    comparison = None
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get("split") != "dev" or not report.get("selection_sha256"):
            raise ValueError(f"{path}: snapshot selection requires a fixed dev set")
        if not str(report.get("vocoder", "")).endswith(".pt"):
            raise ValueError(f"{path}: vocoder must be an explicit snapshot checkpoint")
        if report.get("speaker_reference") != "recording":
            raise ValueError(f"{path}: copy-synthesis needs recorded audio as reference")
        rows = {r.get("system"): r for r in report.get("rows", [])}
        if not {"recording", "ground_truth_vocoded", "drifting_tts"} <= rows.keys():
            raise ValueError(f"{path}: missing recording, copy-synthesis or generated-speech row")
        gen = rows["drifting_tts"]
        settings = (report.get("model"), report.get("split"), report.get("offset"),
                    report.get("num_utterances"), report.get("selection_sha256"),
                    json.dumps(report.get("judges"), sort_keys=True), gen.get("temperature"), gen.get("cfg"))
        if comparison is None:
            comparison = settings
        elif settings != comparison:
            raise ValueError(f"{path}: evaluation settings differ from first report")
        scores = {}
        for kind in ("drifting_tts", "ground_truth_vocoded"):
            per_voice = rows[kind].get("per_speaker", {})
            if set(per_voice) != set(SPEAKERS):
                raise ValueError(f"{path}: {kind} lacks both Malayalam voices")
            values = [per_voice[s].get("wer") for s in SPEAKERS]
            if any(not isinstance(v, (int, float)) or not 0 <= v < float("inf") for v in values):
                raise ValueError(f"{path}: {kind} has invalid per-voice WER")
            scores[kind] = dict(zip(SPEAKERS, values))
        ranked.append({"vocoder": report["vocoder"], "report": path,
                       "generated_wer": scores["drifting_tts"],
                       "copy_wer": scores["ground_truth_vocoded"],
                       "worst_generated_wer": max(scores["drifting_tts"].values()),
                       "worst_copy_wer": max(scores["ground_truth_vocoded"].values())})
    return sorted(ranked, key=lambda r: (r["worst_generated_wer"], r["worst_copy_wer"]))


def run(args: argparse.Namespace) -> None:
    ranked = rank_reports(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ranking": ranked, "note": "dev-only aid; listen to copy and generated speech "
                               "for both voices before choosing"}, indent=2) + "\n")
    print(f"ranked {len(ranked)} vocoder snapshots: {out}")
