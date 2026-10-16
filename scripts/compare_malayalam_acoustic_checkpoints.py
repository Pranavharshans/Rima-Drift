"""Generate same-sentence, no-prosody audio across Malayalam acoustic checkpoints."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer


def write_wav(path: Path, wav: torch.Tensor) -> float:
    pcm = (wav.detach().cpu().clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return round(wav.numel() / SAMPLE_RATE, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, help="step:path pairs")
    p.add_argument("--vocoder", required=True)
    p.add_argument("--diagnostic-manifest", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.diagnostic_manifest).read_text())["items"]
    result = {"config": vars(args), "note": "unbatched FP32, no prosody, same Vocos/text/voice/seed/cfg/temperature", "items": []}
    for spec in args.models:
        step, model_path = spec.split(":", 1)
        synth = Synthesizer(model_path, "cuda", vocoder=args.vocoder,
                            prosody=None, fast=False, compile=False, tf32=False)
        for item in items:
            speaker, text, seed = item["speaker"], item["text"], item["seed"]
            wav, _ = synth(text, speaker=speaker, cfg_scale=2.0,
                           temperature=0.3, seed=seed, pause=0.15)
            filename = f"{speaker}_{step}.wav"
            row = {"step": int(step), "speaker": speaker, "source_id": item["source_id"],
                   "text": text, "seed": seed, "file": filename,
                   "seconds": write_wav(out / filename, wav)}
            result["items"].append(row)
            (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"saved {filename}: {row['seconds']}s", flush=True)
        del synth
        torch.cuda.empty_cache()
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
