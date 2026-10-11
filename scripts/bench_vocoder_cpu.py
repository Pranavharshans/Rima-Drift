"""CPU latency of exported vocoder graphs (``scripts/export_onnx.py --only vocoder``): a stand-in for phones (#8).

For every ``name=path/to/vocoder.onnx`` and thread count, ONNX Runtime on the CPU times
* the first streaming window: ``first`` frames plus the vocoder's right context (``config.json`` next to the graph,
  as ``drifting_tts.fast`` and ``web/tts.js`` stream), i.e. the vocoder's share of the time to first audio;
* a whole utterance of ``--seconds`` seconds, given as the real-time factor (compute time / audio duration).

With ``--mels`` (unnormalised BigVGAN log-mels ``[100, T]`` in ``.npy`` files) the fp16-weight copy of each graph
(``vocoder_fp16.onnx``) is also compared with its fp32 graph: the SNR of its audio.

    python scripts/bench_vocoder_cpu.py vocos-v2=onnx/vocos_v2/vocoder.onnx bigvgan-v2-ft=onnx/bigvgan/vocoder.onnx \\
        --threads 1 2 4 --out vocoder_cpu.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def timed(sess, mel: np.ndarray, repeats: int, warmup: int = 2) -> float:
    """Median wall time of ``repeats`` runs in milliseconds."""
    for _ in range(warmup):
        sess.run(None, {"mel": mel})
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, {"mel": mel})
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    return float(10 * np.log10((ref**2).sum() / max(((ref - x) ** 2).sum(), 1e-20)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("graphs", nargs="+", help="name=path/to/vocoder.onnx")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seconds", type=float, default=5.0, help="length of the whole-utterance run")
    p.add_argument("--first", type=int, default=32, help="frames of the first streaming window (0.34 s)")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--mels", nargs="*", default=[], help=".npy log-mels [100, T] for the fp16 parity check")
    p.add_argument("--out", default=None, help="JSON with every measurement")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    frames = int(round(args.seconds * 24000 / 256))
    rows = []
    for spec in args.graphs:
        name, _, path = spec.partition("=")
        path = Path(path)
        meta = json.loads((path.parent / "config.json").read_text()).get("vocoder", {})
        context = int(meta.get("context") or 32)
        first = (rng.standard_normal((1, 100, args.first + context)) * 2 - 5).astype(np.float32)
        whole = (rng.standard_normal((1, 100, frames)) * 2 - 5).astype(np.float32)
        row = {"name": name, "file": str(path), "mb": path.stat().st_size / 1e6, "context": context,
               "parameters": meta.get("parameters")}
        for th in args.threads:
            sess = session(path, th)
            row[f"first_ms_{th}t"] = timed(sess, first, args.repeats)
            row[f"rtf_{th}t"] = timed(sess, whole, max(3, args.repeats // 3)) / 1e3 / args.seconds
            print(f"{name}: {th} thread(s): first window {row[f'first_ms_{th}t']:.1f} ms, "
                  f"RTF {row[f'rtf_{th}t']:.4f}", flush=True)
        half = path.with_name(path.stem + "_fp16.onnx")
        if args.mels and half.exists():
            s32, s16 = session(path, max(args.threads)), session(half, max(args.threads))
            snrs = []
            for f in args.mels:
                mel = np.load(f).astype(np.float32)[None]
                snrs.append(snr_db(s32.run(None, {"mel": mel})[0], s16.run(None, {"mel": mel})[0]))
            row["fp16_snr_db"] = statistics.median(snrs)
            print(f"{name}: fp16 weights, audio SNR vs fp32 {row['fp16_snr_db']:.1f} dB (median of {len(snrs)})")
        rows.append(row)

    cols = [f"first_ms_{t}t" for t in args.threads] + [f"rtf_{t}t" for t in args.threads]
    print("\n| vocoder | params | file | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for r in rows:
        params = f"{r['parameters'] / 1e6:.1f} M" if r.get("parameters") else "–"
        print(f"| {r['name']} | {params} | {r['mb']:.0f} MB | "
              + " | ".join(f"{r[c]:.1f}" if c.startswith("first") else f"{r[c]:.4f}" for c in cols) + " |")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
