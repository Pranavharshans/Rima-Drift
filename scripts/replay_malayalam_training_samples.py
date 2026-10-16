"""Replay exact acoustic training samples through BigVGAN and the later Vocos."""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.data import MelDataset
from drifting_tts.train import load_tts
from drifting_tts.vocoder import Vocoder, load_vocoder


def write_wav(path: Path, wav: torch.Tensor) -> float:
    wav = wav.detach().cpu().flatten()
    pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    return round(wav.numel() / SAMPLE_RATE, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, help="step:path pairs")
    p.add_argument("--vocos", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--training-samples", required=True)
    p.add_argument("--indices", nargs="+", type=int, default=[1, 2])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = MelDataset(args.data, "val", min_frames=1, max_frames=10**9, with_audio=True)
    bigvgan = Vocoder("cuda", backend=ds.backend)
    vocos = load_vocoder(args.vocos, "cuda", backend=ds.backend)
    result = {"config": vars(args), "note": "exact log_samples text, speaker, seed=i, CFG=2, temperature=0.5, direct model.synthesize; same mel sent to both vocoders", "items": []}
    for i in args.indices:
        item = ds[i]
        source = ds.items[i]
        result["items"].append({"index": i, "source_id": source["source_id"],
                                "speaker": source["speaker"], "text": source["norm_text"],
                                "files": {"recording": f"utt{i}_recording.wav"}})
        write_wav(out / f"utt{i}_recording.wav", item["audio"])
    for spec in args.models:
        step, path = spec.split(":", 1)
        model, _, _ = load_tts(path, "cuda")
        model.eval()
        for i in args.indices:
            item = ds[i]
            text = item["text"][None].to("cuda")
            spk = torch.tensor([item["spk"]], device="cuda")
            g = torch.Generator(device="cuda").manual_seed(i)
            with torch.no_grad():
                mel, _ = model.synthesize(text, torch.tensor([text.shape[1]], device="cuda"), spk,
                                          cfg_scale=2.0, temperature=0.5, generator=g)
                mel = ds.stats.denormalize(mel)
                wave_big = bigvgan(mel)[0]
                wave_vocos = vocos(mel)[0]
            files = {}
            for name, wav in (("bigvgan_replay", wave_big), ("vocos_replay", wave_vocos)):
                filename = f"utt{i}_step{step}_{name}.wav"
                files[name] = {"file": filename, "seconds": write_wav(out / filename, wav)}
            saved = Path(args.training_samples) / f"step{step}_utt{i}_cfg2.wav"
            if not saved.is_file():
                raise FileNotFoundError(saved)
            filename = f"utt{i}_step{step}_saved_training.wav"
            shutil.copyfile(saved, out / filename)
            files["saved_training"] = {"file": filename}
            next(x for x in result["items"] if x["index"] == i)["files"][step] = files
            (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"saved utt{i} step{step}: BigVGAN {files['bigvgan_replay']['seconds']}s, "
                  f"Vocos {files['vocos_replay']['seconds']}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
