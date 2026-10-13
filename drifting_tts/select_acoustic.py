"""Rank Malayalam acoustic checkpoints using balanced development-set evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEAKERS = ("malayalam_female", "malayalam_male")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", nargs="+", required=True, help="evaluate results.json files from the same dev set")
    p.add_argument("--out", required=True)


def rank_reports(paths: list[str]) -> list[dict]:
    ranked = []
    comparison = None
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get("split") != "dev":
            raise ValueError(f"{path}: checkpoint selection must use the dev split")
        if not report.get("selection_sha256"):
            raise ValueError(f"{path}: evaluation is missing the selected-utterance hash")
        rows = [r for r in report.get("rows", []) if r.get("system") == "drifting_tts"]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one fixed temperature/CFG acoustic row, got {len(rows)}")
        row = rows[0]
        settings = (report.get("split"), report.get("offset"), report.get("num_utterances"),
                    report.get("selection_sha256"),
                    report.get("vocoder"), json.dumps(report.get("judges"), sort_keys=True),
                    row.get("temperature"), row.get("cfg"))
        if comparison is None:
            comparison = settings
        elif settings != comparison:
            raise ValueError(f"{path}: evaluation settings differ from the first report")
        per_speaker = row.get("per_speaker", {})
        if set(per_speaker) != set(SPEAKERS):
            raise ValueError(f"{path}: missing Malayalam speaker metrics: {set(SPEAKERS) - set(per_speaker)}")
        wers = []
        for speaker in SPEAKERS:
            wer = per_speaker[speaker].get("wer")
            if not isinstance(wer, (int, float)) or not 0 <= wer < float("inf"):
                raise ValueError(f"{path}: invalid {speaker} WER: {wer!r}")
            wers.append(float(wer))
        ranked.append({"model": report["model"], "report": path, "temperature": row.get("temperature"),
                       "cfg": row.get("cfg"), "per_speaker_wer": dict(zip(SPEAKERS, wers)),
                       "mean_voice_wer": sum(wers) / 2, "worst_voice_wer": max(wers)})
    return sorted(ranked, key=lambda r: (r["worst_voice_wer"], r["mean_voice_wer"]))


def run(args: argparse.Namespace) -> None:
    ranked = rank_reports(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ranking": ranked, "note": "dev-only aid; listen to both voices before selecting"},
                              indent=2) + "\n")
    print(f"ranked {len(ranked)} checkpoints; inspect both voices before choosing: {out}")
