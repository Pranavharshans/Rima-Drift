"""Same-text Malayalam A/B: recording, Vocos, and three unbatched synthesis settings."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer


def write_wav(path: Path, wav: torch.Tensor) -> float:
    wav = wav.detach().cpu().flatten()
    pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return round(wav.numel() / SAMPLE_RATE, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--original-manifest", required=True)
    p.add_argument("--source-ids", nargs="+", default=["MAL_F_SAD_00441", "MAL_M_WIKI_00133"])
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for line in (root / "index.jsonl").open():
        row = json.loads(line)
        if row.get("source_id") in args.source_ids and row.get("split") == "dev":
            rows[row["source_id"]] = row
    if set(rows) != set(args.source_ids):
        raise ValueError(f"missing dev source IDs: {set(args.source_ids) - set(rows)}")
    original_seeds = {item["text"]: item["seed"] for item in
                      json.loads(Path(args.original_manifest).read_text())["clips"]}

    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder, prosody=args.prosody,
                        fast=False, compile=False, tf32=False)
    audio = np.memmap(root / "audio.bin", dtype=np.float16, mode="r")
    mels = np.memmap(root / "mels.bin", dtype=np.float16, mode="r").reshape(-1, 100)
    result = {"config": vars(args), "note": "same text and seed; unbatched FP32 inference; recorded mel is the dataset mel fed through this Vocos checkpoint", "items": []}
    for source_id in args.source_ids:
        row = rows[source_id]
        speaker = row["speaker"].removeprefix("malayalam_")
        stem = "female" if speaker == "female" else "male"
        text = row["norm_text"]
        seed = original_seeds[text]
        item = {"source_id": source_id, "speaker": speaker, "text": text,
                "seed": seed, "files": {}}
        a = np.array(audio[row["audio_offset"]:row["audio_offset"] + row["audio_samples"]], dtype=np.float32)
        item["files"]["recording"] = {"file": f"{stem}_recording.wav",
                                      "seconds": write_wav(out / f"{stem}_recording.wav", torch.from_numpy(a))}
        mel = np.array(mels[row["offset"]:row["offset"] + row["frames"]], dtype=np.float32)
        with torch.no_grad():
            rec = synth.vocoder(torch.from_numpy(mel.T.copy())[None].to("cuda"))[0]
        item["files"]["recorded_mel_vocos"] = {
            "file": f"{stem}_recorded_mel_vocos.wav",
            "seconds": write_wav(out / f"{stem}_recorded_mel_vocos.wav", rec)}

        for mode in ("sampled", "regressor", "none"):
            synth.set_prosody(None if mode == "none" else args.prosody,
                              durations="sampled" if mode == "none" else mode)
            wav, _ = synth(text, speaker=speaker, cfg_scale=2.0,
                           temperature=0.3, seed=seed, pause=0.15)
            filename = f"{stem}_{mode}.wav"
            item["files"][mode] = {"file": filename,
                                   "seconds": write_wav(out / filename, wav)}
        result["items"].append(item)
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved {source_id}: " + ", ".join(
            f"{name}={info['seconds']}s" for name, info in item["files"].items()), flush=True)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
