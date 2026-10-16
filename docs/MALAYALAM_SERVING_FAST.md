# Malayalam `Serving.fast()` benchmark

Measured 16 October 2026 on the same idle RTX 3060 Laptop GPU (12 GB) and
Malayalam checkpoints as [the FP32 concurrency experiment](MALAYALAM_CONCURRENCY.md):
150k acoustic, 110k Vocos continuation, and 20k prosody. The VM ran PyTorch
2.11.0+cu128. Both paths loaded the same FP32 checkpoint files. Fast mode used
`Serving.fast()` (FP16 autocast for text, prosody, DiT, and Vocos; four length
buckets where the batch is large enough; `torch.compile`). The measured stream
settings, seeds, prompts, and output timing boundary matched the previous FP32
concurrency run. Chunked acoustic streaming was disabled in both paths.

Measurements below are **warm**, after each request shape ran once. Compilation
is excluded from the latency figures. For the original-prompt single suite,
the first fast warmup took 132.6 seconds; further shapes took 0.3–1.0 seconds
with the compiler cache warm. The separate concurrency processes spent several
minutes compiling shapes. A service must warm its expected shapes before
accepting latency-sensitive traffic, and new shapes can still trigger compilation.

## Single request, original short/long/paragraph prompts

These are the exact texts from the earlier
[Malayalam GPU single-request report](MALAYALAM_V32_RESULTS.md#prosody-completion-and-gpu-generation-speed).
Each path was warmed once and then run five times. Values are median TTFA and
end-to-end latency in milliseconds. Both columns here use `stream_batched` with
one request so that changing `Serving()` to `Serving.fast()` is the only serving
configuration change. The earlier report used `Synthesizer.stream`, so its
absolute times are a separate reference rather than an A/B baseline.

| Voice / prompt | FP32 TTFA / E2E | Fast TTFA / E2E | TTFA speedup | Audio length |
|---|---:|---:|---:|---:|
| Female short | 60.2 / 125.3 | 36.7 / 79.2 | 1.64× | 2.187 s |
| Female long | 57.9 / 67.2 | 36.4 / 47.3 | 1.59× | 5.141 s |
| Female paragraph | 57.9 / 191.9 | 37.0 / 131.0 | 1.57× | 8.631 s |
| Male short | 57.5 / 121.2 | 36.2 / 77.7 | 1.59× | 2.241 s |
| Male long | 57.1 / 66.5 | 39.0 / 49.9 | 1.46× | 5.088 s |
| Male paragraph | 58.2 / 195.9 | 37.0 / 132.8 | 1.57× | 8.695 s |

The first report's single-request `Synthesizer.stream` TTFA was approximately
73–78 ms with prosody for these streaming cases. It was not rerun here because
that API cannot use `Serving.fast()`; the matched single-request comparison
above isolates the new serving option.

## Simultaneous arrivals

Each shape was warmed once and timed twice. Requests contain the same distinct
Malayalam dev text and every fourth request contains three sentences. TTFA is
arrival until the first 24 kHz PCM reaches the host. `All done` is the time to
finish the entire burst. This is an offline single-GPU simulation, with no HTTP,
transport, encoding, or sustained-arrival load.

| Strategy / requests | FP32 TTFA p50 / p99 | Fast TTFA p50 / p99 | FP32 / fast all done | FP32 / fast GPU reserved | FP32 / fast stalled |
|---|---:|---:|---:|---:|---:|
| One batch / 1 | 61 / 61 ms | 30 / 31 ms | 73 / 40 ms | 0.37 / 0.37 GiB | 0 / 0% |
| One batch / 8 | 85 / 88 ms | 49 / 60 ms | 401 / 252 ms | 0.45 / 0.44 GiB | 0 / 0% |
| One batch / 16 | 195 / 199 ms | 75 / 83 ms | 797 / 356 ms | 0.63 / 0.51 GiB | 0 / 0% |
| One batch / 32 | 517 / 518 ms | 182 / 197 ms | 1,475 / 595 ms | 1.20 / 0.79 GiB | 0 / 0% |
| One batch / 64 | 1,041 / 1,041 ms | 327 / 330 ms | 2,460 / 890 ms | 2.12 / 1.42 GiB | 0 / 0% |
| One batch / 100 | 1,856 / 1,862 ms | 585 / 605 ms | 3,960 / 1,369 ms | 3.79 / 2.20 GiB | 1 / 0% |
| One batch / 200 | 3,685 / 3,689 ms | 645 / 656 ms | 8,223 / 2,352 ms | 7.60 / 3.50 GiB | 99 / 0% |
| Groups of 64 / 100 | 1,045 / 3,121 ms | 341 / 1,107 ms | 4,079 / 1,509 ms | 1.96 / 1.38 GiB | 0 / 0% |
| Groups of 64 / 200 | 3,625 / 8,692 ms | 1,294 / 2,949 ms | 8,892 / 3,188 ms | 2.65 / 1.62 GiB | 0 / 0% |

At 200 arrivals, one batch delivered approximately 712 seconds of generated
audio per wall-clock second in fast mode, versus 204 in FP32. The two outputs
contained 1,675.10 and 1,675.07 seconds of audio respectively. For groups of
64, fast mode gave lower peak GPU reserved memory but higher process RAM (2.66
versus 1.51 GiB at 200). The compiled process needs memory headroom beyond its
GPU peak. FP32's modeled playback stall at 200 one-batch arrivals did not occur
in this fast run; this is not a production stall guarantee.

## Audio comparison and interpretation

The six FP32/fast WAV pairs are in
[`docs/benchmarks/malayalam-serving-fast/samples/`](benchmarks/malayalam-serving-fast/samples/).
Each pair has exactly the same sample count. Across the pairs, sample-aligned
waveform correlation was 0.9973–0.9995 and signal-to-difference ratio was
22.6–30.0 dB. These numerical checks show small waveform changes on these
prompts, **not** that a listener will hear no quality difference. Listen to
both voices, especially pronunciation and artifacts, before choosing fast mode
for deployment. This test did not measure intelligibility or a held-out audio
quality metric.

`Serving.fast()` does **not** shrink the checkpoint files: it runs FP32 weights
under mixed-precision autocast. Nor do these warm offline figures establish
HTTP latency, sustainable concurrent-user capacity, or performance on a 40/50
series GPU. Rebenchmark on the intended deployment GPU and server stack.

Reproduce with [`bench_malayalam_concurrency.py`](../scripts/bench_malayalam_concurrency.py)
using `--fast-serving`, and
[`bench_malayalam_single_fast.py`](../scripts/bench_malayalam_single_fast.py).
Raw JSON: [one batch](benchmarks/malayalam-serving-fast/batched_fast.json),
[groups of 64](benchmarks/malayalam-serving-fast/microbatch64_fast.json), and
[single prompts](benchmarks/malayalam-serving-fast/single_results.json).
