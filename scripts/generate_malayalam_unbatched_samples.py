"""Generate individual Malayalam WAVs through the standard, unbatched FP32 path."""

from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer


def candidates(index: Path) -> list[str]:
    texts = list(dict.fromkeys(
        str(row.get("norm_text", "")).strip()
        for line in index.open()
        if (row := json.loads(line)).get("split") == "dev"
    ))
    texts = [text for text in texts
             if 90 <= len(text) <= 170
             and sum("\u0d00" <= char <= "\u0d7f" for char in text) / len(text) >= 0.7]
    random.Random(20261016).shuffle(texts)
    return texts


def write_wav(path: Path, wav: torch.Tensor) -> None:
    pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=5)
    p.add_argument("--min-seconds", type=float, default=4.0)
    args = p.parse_args()
    if args.clips < 1 or args.min_seconds <= 0:
        p.error("--clips and --min-seconds must be positive")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder, prosody=args.prosody,
                        fast=False, compile=False, tf32=False)
    result = {"config": vars(args), "gpu": torch.cuda.get_device_name(0),
              "path": "Synthesizer.__call__, one request at a time, FP32, no CUDA graphs or compile",
              "clips": [], "rejected_short": 0}
    for candidate_id, text in enumerate(candidates(Path(args.index))):
        if len(result["clips"]) >= args.clips:
            break
        pair = []
        for speaker in ("female", "male"):
            wav, timing = synth(text, speaker=speaker, cfg_scale=2.0,
                                 temperature=0.3, seed=1000 + candidate_id, pause=0.15)
            pair.append((speaker, wav, timing))
        if min(item[2]["seconds"] for item in pair) < args.min_seconds:
            result["rejected_short"] += 1
            continue
        number = len(result["clips"]) + 1
        clip = {"number": number, "text": text, "seed": 1000 + candidate_id,
                "voices": {}}
        for speaker, wav, timing in pair:
            name = f"{number:02d}_{speaker}.wav"
            write_wav(out / name, wav)
            clip["voices"][speaker] = {"file": name, "seconds": round(timing["seconds"], 3),
                                       "rtf_total": round(timing["rtf_total"], 4)}
        result["clips"].append(clip)
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved {number}/{args.clips}: " + ", ".join(
            f"{speaker}={timing['seconds']:.2f}s" for speaker, _, timing in pair), flush=True)
    if len(result["clips"]) != args.clips:
        raise RuntimeError(f"only found {len(result['clips'])} clips lasting {args.min_seconds}+ seconds")
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
