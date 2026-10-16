"""Small no-prosody CFG/temperature sweep for Malayalam articulation listening."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer


SETTINGS = ((1.0, 0.3), (2.0, 0.3), (3.0, 0.3), (2.0, 0.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--diagnostic-manifest", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.diagnostic_manifest).read_text())["items"]
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder,
                        prosody=None, fast=False, compile=False, tf32=False)
    result = {"config": vars(args), "note": "unbatched FP32; acoustic/Vocos/text/voice/seed fixed; no prosody", "items": []}
    for item in items:
        for cfg, temperature in SETTINGS:
            wav, _ = synth(item["text"], speaker=item["speaker"], seed=item["seed"],
                           cfg_scale=cfg, temperature=temperature, pause=0.15)
            filename = f"{item['speaker']}_cfg{cfg:g}_temp{temperature:g}.wav"
            pcm = (wav.detach().cpu().clamp(-1, 1) * 32767).to(torch.int16).numpy()
            with wave.open(str(out / filename), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SAMPLE_RATE)
                f.writeframes(pcm.tobytes())
            row = {"speaker": item["speaker"], "source_id": item["source_id"],
                   "text": item["text"], "seed": item["seed"], "cfg_scale": cfg,
                   "temperature": temperature, "file": filename,
                   "seconds": round(wav.numel() / SAMPLE_RATE, 3)}
            result["items"].append(row)
            (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"saved {filename}: {row['seconds']}s", flush=True)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
