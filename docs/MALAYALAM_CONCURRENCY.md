# Malayalam GPU concurrency experiment

Measured on 16 October 2026 after the 20k-step prosody run finished. This is a
**single-GPU serving simulation**, not an HTTP load test or a claim about a
deployed service. It uses the 150k acoustic checkpoint, 110k Vocos continuation
checkpoint and `prosody_ema.pt` on an otherwise idle NVIDIA RTX 3060 Laptop GPU
(12 GB). The benchmarks use the repository's streaming and batched inference
paths, fp32, one shared loaded model and no CUDA graphs or compilation.

Each request contains different Malayalam dev text. Every fourth request is
three sentences; the others are one sentence. All requests in a row arrive at
the same instant. Each shape was warmed once, then measured twice with the same
texts and seeds. The tables are female-voice bursts unless labelled male.
The 100-request mix produces about 816 seconds of audio; the 200-request mix
about 1,675 seconds. These are offline accelerated generation tests, so the
sum of generated audio can greatly exceed wall-clock time.

TTFA is arrival to the first 24 kHz PCM piece on the host. End-to-end (E2E)
latency is arrival to the last PCM piece; it includes queueing. Per-request RTF
is E2E divided by that request's audio duration, so it also includes queueing.
An RTF above 1 means the request completed slower than its audio duration.
`All done` is the burst makespan. The playback-stall estimate starts playback
when each request's first piece arrives, then checks whether later pieces arrive
in time. Transport, HTTP, encoding, network latency, client playback and model
load are excluded. This is not a sustained-arrival or multi-tenant test.

## At 100 and 200 simultaneous arrivals

All times are milliseconds. GPU memory is peak PyTorch **reserved**, which
includes its allocator cache; RAM is process high-water resident memory.

| Voice | Requests | Strategy | TTFA p50 / p99 | E2E p50 / p99 | Request RTF p50 | All done | Playback stalled | GPU / RAM GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Female | 100 | FIFO | 6,860 / 13,661 | 6,872 / 13,666 | 0.96 | 13,954 | 0% | 0.40 / 1.47 |
| Female | 100 | One batch | 1,856 / 1,862 | 2,716 / 3,959 | 0.44 | 3,960 | 1% | 3.79 / 1.58 |
| Female | 100 | Groups of 32 | 1,947 / 4,282 | 2,278 / 4,413 | 0.31 | 4,480 | 0% | 1.50 / 1.46 |
| Female | 100 | Groups of 64 | 1,045 / 3,121 | 2,267 / 4,096 | 0.33 | 4,079 | 0% | 1.96 / 1.50 |
| Female | 200 | FIFO | 14,118 / 29,826 | 14,162 / 29,992 | 2.13 | 29,898 | 0% | 0.40 / 1.48 |
| Female | 200 | One batch | 3,685 / 3,689 | 5,577 / 8,134 | 0.87 | 8,223 | **99%** | 7.60 / 1.65 |
| Female | 200 | Groups of 32 | 4,626 / 8,669 | 4,742 / 8,954 | 0.67 | 9,079 | 0% | 1.51 / 1.47 |
| Female | 200 | Groups of 64 | 3,625 / 8,692 | 4,119 / 8,944 | 0.64 | 8,892 | 0% | 2.65 / 1.51 |
| Male | 100 | One batch | 1,639 / 1,648 | 2,426 / 3,618 | 0.44 | 3,597 | 1% | 2.83 / 1.51 |
| Male | 100 | Groups of 64 | 884 / 2,862 | 2,041 / 3,786 | 0.31 | 3,787 | 1% | 1.78 / 1.50 |
| Male | 200 | One batch | 3,230 / 3,231 | 4,875 / 7,248 | 0.82 | 7,304 | **98.5%** | 5.71 / 1.61 |
| Male | 200 | Groups of 64 | 3,359 / 7,390 | 3,791 / 7,689 | 0.65 | 7,818 | 0.5% | 2.38 / 1.51 |

One large batch gives nearly identical TTFA to every request, but its vocoder
rounds become too slow for uninterrupted streaming around 200 requests. Groups
of 64 reduce median TTFA at 100 requests and keep stalls rare through 200;
their last group must wait in a queue, which raises p99 TTFA. Groups of 32
use less GPU memory but have somewhat higher queue latency. FIFO uses little
memory but has poor first-audio latency under a burst.

## Ramp to memory failure and larger queues

| Female requests | One-batch TTFA p50 / stalls / reserved GiB | Groups-of-64 TTFA p50 / stalls / reserved GiB | Groups-of-32 TTFA p50 / stalls / reserved GiB |
|---:|---:|---:|---:|
| 32 | 517 ms / 0% / 1.20 | — | 516 ms / 0% / 1.16 |
| 64 | 1,041 ms / 0% / 2.12 | 1,041 ms / 0% / 1.96 | — |
| 128 | 2,368 ms / 1.6% / 5.18 | — | — |
| 256 | 4,713 ms / 99.2% / 10.37 | — | — |
| 384 | 7,065 ms / 99.5% / 9.40 | 7,598 ms / 0.1% / 2.65 | 8,357 ms / 0% / 1.51 |
| 512 | **CUDA out of memory** | 10,119 ms / 0% / 2.65 | 11,168 ms / 0% / 1.51 |
| 768 | — | — | 16,488 ms / 0% / 1.51 |

The one-batch 512-request attempt raised a caught CUDA OOM: the process used
about 11.05 GiB of the 11.63 GiB available when it attempted another 1.78 GiB
allocation. The GPU remained usable afterward. The per-shape reserved memory
is not strictly monotonic because padding, temporary allocations and caching
vary; 384 succeeding does **not** establish a reliable 384-request production
limit. The 200-request playback-stall result is already a more restrictive
limit for an uninterrupted stream.

At 100–200 requests, batches of 64 delivered roughly 189–200 seconds of audio
per wall-clock second, about 22–26 completed requests per second for this mix
of roughly eight-second clips. That is a burst-completion rate, **not** a
measured sustainable arrival rate or a capacity guarantee. A hosted service
needs a queue cap, timeout/backpressure policy, request-length limits, and an
HTTP load test with network and encoding included. In particular, 100 arrivals
already give a 2.9–3.1 second p99 TTFA with groups of 64; 200 arrivals give
7.4–8.7 seconds p99. This GPU is not a low-latency 100–200-user-at-once host.

The reproducible script is [`scripts/bench_malayalam_concurrency.py`](../scripts/bench_malayalam_concurrency.py).
Raw summaries, including all reported percentiles and per-case allocated and
reserved memory, are in [`docs/benchmarks/malayalam-concurrency/`](benchmarks/malayalam-concurrency/).
Full remote run outputs are under `/workspace/Rima-Drift/runs/concurrency_ml/`.
