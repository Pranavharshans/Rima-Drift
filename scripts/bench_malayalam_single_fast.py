"""Compare FP32 and Serving.fast() on the original Malayalam single-request prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.batched import Serving, stream_batched
from drifting_tts.synthesize import Synthesizer


def save_wav(path: Path, samples: torch.Tensor) -> None:
    pcm = (samples.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def run(synth: Synthesizer, text: str, speaker: str, serving: Serving):
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = None
    pieces = []
    for batch in stream_batched(synth, [text], speaker=speaker, seeds=[0],
                                cfg_scale=2.0, temperature=0.3, pause=0.15,
                                chunked=None, serving=serving):
        if first is None:
            first = time.perf_counter() - start
        pieces.extend(piece for _, piece in batch)
    end = time.perf_counter() - start
    wav = torch.cat(pieces)
    return first, end, wav


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--original-results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = json.loads(Path(args.original_results).read_text())["texts"]
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder, prosody=args.prosody)
    data = {"config": vars(args), "gpu": torch.cuda.get_device_name(0), "results": {}}
    for speaker in ("female", "male"):
        for name in ("short", "long", "paragraph"):
            text = texts[name]
            for label, serving in (("fp32", Serving()), ("fast", Serving.fast())):
                start = time.perf_counter()
                run(synth, text, speaker, serving)
                warmup = time.perf_counter() - start
                rows = [run(synth, text, speaker, serving) for _ in range(args.repeats)]
                ttfa = statistics.median(r[0] for r in rows)
                e2e = statistics.median(r[1] for r in rows)
                wav = rows[-1][2]
                save_wav(out / f"{speaker}_{name}_{label}.wav", wav)
                result = {"warmup_s": round(warmup, 3), "ttfa_ms": round(ttfa * 1000, 2),
                          "e2e_ms": round(e2e * 1000, 2),
                          "audio_s": round(wav.numel() / SAMPLE_RATE, 3),
                          "rtf": round(e2e / (wav.numel() / SAMPLE_RATE), 4),
                          "peak": round(wav.abs().max().item(), 4)}
                data["results"][f"{speaker}_{name}_{label}"] = result
                (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
                print(f"{speaker} {name} {label}: {result}", flush=True)


if __name__ == "__main__":
    main()
