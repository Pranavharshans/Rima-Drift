"""Compare predicted and reference MAS duration/pitch on two held-out Rasa clips."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import torch

from drifting_tts.alignment import sequence_mask
from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.data import MelDataset
from drifting_tts.models.text_encoder import align, token_pitch
from drifting_tts.train import load_tts
from drifting_tts.vocoder import load_vocoder


def save(path: Path, wav: torch.Tensor) -> float:
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
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-ids", nargs="+", default=["MAL_F_SAD_00441", "MAL_M_WIKI_00133"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = MelDataset(args.data, "dev", min_frames=1, max_frames=10**9, with_f0=True, with_audio=True)
    indices = {row["source_id"]: i for i, row in enumerate(ds.items) if row.get("source_id") in args.source_ids}
    if set(indices) != set(args.source_ids):
        raise ValueError(f"Missing dev IDs: {set(args.source_ids) - set(indices)}")
    model, _, _ = load_tts(args.model, "cuda")
    vocoder = load_vocoder(None, "cuda", backend=ds.backend)
    result = {"config": vars(args), "note": "Reference durations are checkpoint MAS alignments, not hand-labelled phonemes. Both reference conditions use dev recordings; they are diagnostic oracle inputs, not deployable TTS inputs.", "items": []}
    for source_id in args.source_ids:
        i = indices[source_id]
        item = ds[i]
        row = ds.items[i]
        text = item["text"][None].cuda()
        text_len = torch.tensor([text.shape[1]], device="cuda")
        spk = torch.tensor([item["spk"]], device="cuda")
        mel_ref = item["mel"][None].cuda()
        f0_ref = item["f0"][None].cuda()
        y_len = torch.tensor([mel_ref.shape[-1]], device="cuda")
        with torch.no_grad():
            _, mu, logw_pred, x_mask = model.encoder(text, text_len, spk)
            attn, _ = align(mu, x_mask, mel_ref, sequence_mask(y_len, mel_ref.shape[-1])[:, None].float())
            durations_ref = attn.sum(-1).long()
            pitch_ref = token_pitch(f0_ref, attn, model.lf0_stats[0], model.lf0_stats[1])
            predicted_frames = (torch.ceil(torch.exp(logw_pred)) * x_mask).sum().item()
            reference_frames = durations_ref.sum().item()
            stem = "female" if "_F_" in source_id else "male"
            files = {"recording": {"file": f"{stem}_recording.wav", "seconds": save(out / f"{stem}_recording.wav", item["audio"])}}
            for use_duration, use_pitch in ((False, False), (True, False), (False, True), (True, True)):
                label = ("reference_duration" if use_duration else "predicted_duration") + "_" + ("reference_pitch" if use_pitch else "predicted_pitch")
                g = torch.Generator(device="cuda").manual_seed(args.seed)
                generated, generated_lengths = model.synthesize(
                    text, text_len, spk, cfg_scale=2.0, temperature=0.5, generator=g,
                    durations=durations_ref if use_duration else None,
                    pitch=pitch_ref if use_pitch else None)
                wav = vocoder(ds.stats.denormalize(generated))[0]
                filename = f"{stem}_{label}.wav"
                files[label] = {"file": filename, "seconds": save(out / filename, wav),
                                "mel_frames": int(generated_lengths[0].item())}
        result["items"].append({"source_id": source_id, "speaker": row["speaker"], "text": row["norm_text"],
                                "reference_mel_frames": int(reference_frames),
                                "predicted_mel_frames": int(predicted_frames), "files": files})
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{source_id}: reference {reference_frames:.0f} frames, predicted {predicted_frames:.0f} frames", flush=True)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
