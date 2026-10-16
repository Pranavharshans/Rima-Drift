"""Simulate simultaneous Malayalam requests on one GPU, without an HTTP server.

Uses the same streaming implementations as bench_concurrency.py. Run one strategy
per process so a CUDA OOM in a large batch cannot invalidate the other results.
All timings start when the requests arrive together and end when PCM reaches the
host; transport, encoding and playback are excluded.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import statistics
import time
from pathlib import Path

import torch

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.batched import Serving
from drifting_tts.synthesize import Synthesizer
from scripts.bench_concurrency import playback, run_batched, run_fifo


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def dev_texts(index: Path, count: int, paragraph_every: int) -> list[str]:
    source = [json.loads(line) for line in index.open()]
    source = list(dict.fromkeys(str(row["norm_text"]).strip() for row in source
                                if row.get("split") == "dev" and str(row.get("norm_text", "")).strip()))
    random.Random(2026).shuffle(source)
    needed = count + (2 * count // paragraph_every if paragraph_every else 0)
    if len(source) < needed:
        raise ValueError(f"need {needed} distinct dev sentences; found {len(source)}")
    out, k = [], 0
    for i in range(count):
        n = 3 if paragraph_every and i % paragraph_every == paragraph_every - 1 else 1
        items = source[k:k + n]
        k += n
        out.append(" ".join(t if t.endswith((".", "?", "!")) else t + "." for t in items))
    return out


def summarize(runs, peak_alloc, peak_reserved, peak_ram):
    requests = len(runs[0])
    ttfa = [ev[0][0] * 1000 for run in runs for ev in run]
    e2e = [ev[-1][0] * 1000 for run in runs for ev in run]
    audio = [sum(n for _, n in ev) / SAMPLE_RATE for run in runs for ev in run]
    stalls = [playback(ev)[0] * 1000 for run in runs for ev in run]
    rtf = [e / 1000 / a for e, a in zip(e2e, audio)]
    all_done = [max(ev[-1][0] for ev in run) for run in runs]
    total_audio = [sum(sum(n for _, n in ev) for ev in run) / SAMPLE_RATE for run in runs]
    def pct(xs):
        return {key: round(percentile(xs, q), 2) for key, q in
                (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))}
    return {"requests": requests, "repeats": len(runs), "ttfa_ms": pct(ttfa),
            "e2e_ms": pct(e2e), "request_rtf": pct(rtf),
            "all_done_ms": round(statistics.median(all_done) * 1000, 2),
            "total_audio_s": round(statistics.median(total_audio), 2),
            "audio_seconds_per_second": round(statistics.median(
                a / d for a, d in zip(total_audio, all_done)), 2),
            "playback_stalled_share": round(sum(x > 0 for x in stalls) / len(stalls), 3),
            "playback_stall_p99_ms": round(percentile(stalls, .99), 2),
            "gpu_peak_allocated_gib": round(max(peak_alloc), 3),
            "gpu_peak_reserved_gib": round(max(peak_reserved), 3),
            "process_peak_ram_gib": round(max(peak_ram), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--vocoder", required=True)
    p.add_argument("--prosody", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--speaker", choices=("female", "male"), default="female")
    p.add_argument("--strategy", choices=("fifo", "batched", "microbatch"), required=True)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--requests", type=int, nargs="+", default=[1, 8, 16, 32, 64, 100, 128, 200, 256, 384])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--paragraph-every", type=int, default=4)
    args = p.parse_args()
    if any(n < 1 for n in args.requests) or args.repeats < 1:
        p.error("requests and repeats must be positive")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    texts = dev_texts(Path(args.index), max(args.requests), args.paragraph_every)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    synth = Synthesizer(args.model, "cuda", vocoder=args.vocoder, prosody=args.prosody)
    model_load = time.perf_counter() - start
    kw = {"seeds": list(range(len(texts))),
          "stream": {"speaker": args.speaker, "cfg_scale": 2.0, "temperature": 0.3,
                     "pause": 0.15, "chunked": None}, "serving": Serving()}
    result = {"config": vars(args), "gpu": torch.cuda.get_device_name(0),
              "gpu_total_gib": round(torch.cuda.mem_get_info()[1] / 2**30, 3),
              "model_load_s": round(model_load, 3), "timing": "simultaneous arrival to CPU PCM",
              "transport": "excluded", "results": {}, "stop": None}

    def save():
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"loaded {result['gpu']} in {model_load:.2f}s; strategy={args.strategy}", flush=True)
    save()
    for n in args.requests:
        selected = texts[:n]
        def run():
            if args.strategy == "fifo":
                return run_fifo(synth, selected, kw)
            return run_batched(synth, selected, kw,
                               args.micro_batch if args.strategy == "microbatch" else None)
        try:
            run()  # Warm this shape and allocator before measuring.
            runs, alloc, reserved, ram = [], [], [], []
            for _ in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                runs.append(run())
                alloc.append(torch.cuda.max_memory_allocated() / 2**30)
                reserved.append(torch.cuda.max_memory_reserved() / 2**30)
                ram.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            row = summarize(runs, alloc, reserved, ram)
            result["results"][str(n)] = row
            print(f"N={n} TTFA={row['ttfa_ms']['p50']}/{row['ttfa_ms']['p99']}ms "
                  f"E2E={row['e2e_ms']['p50']}/{row['e2e_ms']['p99']}ms "
                  f"all={row['all_done_ms']}ms GPU={row['gpu_peak_reserved_gib']}GiB "
                  f"RAM={row['process_peak_ram_gib']}GiB", flush=True)
            save()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            result["stop"] = {"requests": n, "reason": "CUDA out of memory", "detail": str(exc)[:300]}
            print(f"stopped at N={n}: CUDA out of memory", flush=True)
            save()
            break
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
