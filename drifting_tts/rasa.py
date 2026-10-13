"""Remote-only Rasa Malayalam ingestion into auditable, split canonical parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED = {"filename", "text", "language", "gender", "duration", "audio"}
SPEAKERS = {"female": "malayalam_female", "male": "malayalam_male"}
OUTPUT_SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()), ("speaker", pa.string()), ("source_id", pa.string()),
    ("source_split", pa.string()), ("duration", pa.float32()), ("style", pa.string()),
])


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="remote output directory; contains gated audio")
    p.add_argument("--revision", required=True, help="pinned ai4bharat/Rasa commit SHA")
    p.add_argument("--source", default=None, help="existing remote Rasa snapshot; skip Hub download")
    p.add_argument("--dev-percent", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inspect-only", action="store_true", help="inspect schemas without writing audio")


def split_for(source_split: str, speaker: str, source_id: str, seed: int, dev_percent: int) -> str:
    if source_split == "test":
        return "val"
    key = f"{seed}:{speaker}:{source_id}".encode()
    return "dev" if int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100 < dev_percent else "train"


def canonicalize(row: dict, source_split: str, seed: int, dev_percent: int) -> tuple[dict | None, str | None]:
    if str(row.get("language", "")).strip().lower() != "malayalam":
        return None, "wrong_language"
    gender = str(row.get("gender", "")).strip().lower()
    if gender not in SPEAKERS:
        return None, "unknown_gender"
    source_id = str(row.get("filename") or "").strip()
    text = str(row.get("text") or "").strip()
    if not source_id or not text:
        return None, "missing_id_or_text"
    try:
        duration = float(row.get("duration"))
    except (TypeError, ValueError):
        return None, "invalid_duration"
    if not 0.5 <= duration <= 30:
        return None, "duration_out_of_range"
    audio = row.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes) or not audio["bytes"]:
        return None, "missing_audio_bytes"
    speaker = SPEAKERS[gender]
    result = {"audio": {"bytes": audio["bytes"], "path": None}, "text": text,
              "speaker": speaker, "source_id": source_id, "source_split": source_split,
              "duration": duration, "style": str(row.get("style") or "")}
    return result, split_for(source_split, speaker, source_id, seed, dev_percent)


def source_files(root: Path) -> list[tuple[Path, str]]:
    files = []
    for split in ("train", "test"):
        files += [(p, split) for p in sorted((root / "Malayalam").glob(f"{split}-*.parquet"))]
    if not files or not any(split == "train" for _, split in files) or not any(split == "test" for _, split in files):
        raise FileNotFoundError(f"expected Malayalam/train-* and Malayalam/test-* parquet files under {root}")
    return files


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.dev_percent <= 20:
        raise ValueError("--dev-percent must be between 1 and 20")
    if args.source:
        root = Path(args.source)
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download("ai4bharat/Rasa", repo_type="dataset", revision=args.revision,
                                      allow_patterns=["Malayalam/train-*.parquet", "Malayalam/test-*.parquet"]))
    files = source_files(root)
    schemas = {}
    for path, _ in files:
        schema = pq.ParquetFile(path).schema_arrow
        missing = REQUIRED - set(schema.names)
        if missing:
            raise ValueError(f"{path}: missing required fields {sorted(missing)}; found {schema.names}")
        audio_type = schema.field("audio").type
        if not pa.types.is_struct(audio_type) or "bytes" not in [field.name for field in audio_type]:
            raise ValueError(f"{path}: expected audio struct with bytes; found {audio_type}")
        schemas[str(path.relative_to(root))] = str(schema)
    if args.inspect_only:
        print(json.dumps({"revision": args.revision, "files": schemas}, indent=2))
        return

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    writers = {split: pq.ParquetWriter(out / f"{split}.parquet", OUTPUT_SCHEMA, compression="zstd")
               for split in ("train", "dev", "val")}
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    try:
        for path, source_split in files:
            pf = pq.ParquetFile(path)
            columns = sorted(REQUIRED | ({"style"} & set(pf.schema_arrow.names)))
            for rg in range(pf.num_row_groups):
                batches = {split: [] for split in writers}
                for row in pf.read_row_group(rg, columns=columns).to_pylist():
                    record, split = canonicalize(row, source_split, args.seed, args.dev_percent)
                    if record is None:
                        counts[f"reject:{split}"] += 1
                        continue
                    key = f"{record['speaker']}:{record['source_id']}"
                    audio_hash = hashlib.sha256(record["audio"]["bytes"]).hexdigest()
                    if key in seen_ids or audio_hash in seen_audio:
                        counts["reject:duplicate_id_or_audio"] += 1
                        continue
                    seen_ids.add(key)
                    seen_audio.add(audio_hash)
                    batches[split].append(record)
                    counts[f"{split}:{record['speaker']}"] += 1
                for split, rows in batches.items():
                    if rows:
                        writers[split].write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
    finally:
        for writer in writers.values():
            writer.close()
    if any(counts[f"{split}:{speaker}"] == 0 for split in ("train", "dev", "val")
           for speaker in SPEAKERS.values()):
        raise ValueError(f"empty speaker/split; inspect audit and source: {dict(counts)}")
    audit = {"dataset": "ai4bharat/Rasa", "revision": args.revision, "seed": args.seed,
             "dev_percent": args.dev_percent, "source_files": [str(p.relative_to(root)) for p, _ in files],
             "counts": dict(sorted(counts.items())), "speaker_map": SPEAKERS,
             "note": "val preserves source test; duplicates removed by speaker+filename or exact audio bytes"}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
