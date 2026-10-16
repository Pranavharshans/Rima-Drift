# Malayalam v3.2 training results and status

Status updated **16 October 2026**. Training stages finished on a remote
NVIDIA RTX 3060 Laptop GPU (12 GB); final quality validation remains. Rasa
data, checkpoints, logs and WAV files remain on that machine; they are not
stored in Git. The original training runs used revision `417735f`; later
inference benchmarking includes the Malayalam streaming fix in `b67dd74`.

## Data and completed components

| Component | Result | Evidence and status |
|---|---|---|
| Rasa Malayalam preparation | 32,466 usable utterances, approximately 49.3 hours, female and male speakers | Prepared remote dataset; approximately 12 GB. |
| Mel-MAE | Completed 60,000 steps | Final `runs/mae2d_ml/mae_ema.pt`; final reported reconstruction loss about 0.042. |
| Acoustic TTS | Completed 150,000 steps in 30.92 hours | The final calibrated EMA checkpoint was selected for subsequent stages: `runs/tts_v3_ml/model_ema_150000.pt`. SHA-256: `2bfa4a59057e9564ab723ab6e047fa5c81c1c061ef44d2b12f58667db4488d76`. |
| Initial Vocos | Completed 40,000 steps | `runs/vocos_ml/last.pt` (full continuation state; SHA-256 `4bdf157e988c49ee13928130b179e6b81df03d95f97e61a213ce94f174355b73`); exported generator `runs/vocos_ml/vocos_ft_40000.pt`. |
| Vocos continuation | Stopped after approximately 117,000 additional steps | Started from the full step-40,000 Vocos state; 110k and 115k exported checkpoints and full 115k state preserved. The 110k export was used for prosody and latency experiments. |
| Stochastic prosody | Completed 20,000 steps | `runs/prosody_ml/prosody_ema.pt`; dev metrics and GPU latency below. |
| Final held-out validation and release packaging | Not started | Required before any quality or release claim. |

The acoustic run ended with reported loss 2.946, `centroid_mse` 0.07187,
`across_sample_std` 0.2028 and zero unaligned tokens at step 150,000. These
training values show that the run completed and alignment did not fail; they do
not measure perceived speech quality.

## Listening observations

These are **informal human comparisons**, not blinded ratings or held-out scores.

- During acoustic training with the earlier BigVGAN output, the 10k sample was
  robotic and jittery but recognizably Malayalam. The 20k and 30k samples grew
  clearer; by 70k, speech was more understandable but still noisy. The final
  150k acoustic checkpoint was chosen for the Vocos runs.
- At initial Vocos step 40k, the original recording was best. The order of
  `recorded_mel` and `gta` varied by utterance. Both reconstructed paths could
  contain jitter, suggesting that the shared mel-to-waveform path contributes
  to the artifact. This is a working diagnosis, not proof of a single cause.
- At continuation step 5k, sample 1's GTA clip sounded cleaner than at the
  initial Vocos 40k checkpoint, with almost no reported jitter; its
  `recorded_mel` clip was slightly better. Sample 0 changed little.
- At continuation step 30k versus continuation step 5k, sample 0's GTA clip
  had less jitter and its `recorded_mel` clip sounded better. Sample 1's GTA
  clips were nearly jitter-free at both steps, and its `recorded_mel` clips
  were nearly noise-free. This improvement supported continuing stage 5.

The comparison WAVs are on the remote machine:

```text
/workspace/Rima-Drift/runs/vocos_ml/listen_step40000/
/workspace/Rima-Drift/runs/vocos_v2_ml/listen_step5000/
/workspace/Rima-Drift/runs/vocos_v2_ml/listen_step30000/
```

Each directory contains `real_0.wav`, `recorded_mel_0.wav`, `gta_0.wav` and
the corresponding `_1.wav` files. `real` is the recording, `recorded_mel` is
Vocos reconstruction of its mel, and `gta` is Vocos reconstruction of an
acoustic-model mel with reference alignment. GTA is not a full free-text
synthesis test. The two `real` WAVs were verified byte-identical between the
initial 40k and continuation 5k comparison sets.

## Evaluation limits and next decisions

An early balanced dev smoke test with Whisper large-v3 produced implausible
Malayalam error rates **even on the original recordings** (recording CER 91.88%,
WER 120.91%). Its outputs included wrong-script transcriptions despite an
explicit Malayalam language setting. Those CER/WER numbers cannot rank model
checkpoints or establish intelligibility. A working Malayalam transcription
evaluation and human listening on both voices are still required.

1. Compare sampled prosody with deterministic timing and pitch for both voices;
   keep the deterministic path if prosody harms either voice.
2. Confirm the 110k Vocos selection with final matched-utterance listening and
   a valid Malayalam intelligibility evaluation.
3. Run final two-voice listening and held-out checks on the exact assembled
   model. Record dataset revision, checkpoint hashes and validated metrics
   before calling the Malayalam model released.

The Vocos continuation log is `/workspace/vocos_v2_ml_full.log`; its output
directory is `/workspace/Rima-Drift/runs/vocos_v2_ml/`.

## Update: 16 October 2026

The Vocos continuation was stopped after approximately 117k steps because a
paired listening set of five new dev utterances per voice showed little audible
difference between 30k and 115k on most clips. The 110k and 115k exported
checkpoints and full 115k state were preserved. The 110k snapshot is a candidate,
not a final quality selection. The small pronunciation error heard only in a GTA
clip is more likely upstream of Vocos; that attribution still needs evaluation.

Prosody targets were cached from the selected 150k acoustic model for 19,635
training and 1,456 dev utterances. A 200-step prosody pilot completed and saved a
checkpoint. The full 20k-step prosody run subsequently completed; its final
metrics are below, and its log is `/workspace/prosody_ml_full.log`.

### CPU generation speed before prosody

The current acoustic checkpoint plus `runs/vocos_v2_ml/vocos_ft_110000.pt` was
benchmarked on the remote Intel Xeon E5-2697A v4 CPU with a 15.36-CPU container
quota. The process used four PyTorch threads at low priority while prosody GPU
training was active. Five warm runs per voice and condition were measured on
Malayalam dev text: short (23 characters), long (88) and a three-sentence item
(125). Values below are medians. These are **pre-prosody** timings, not final
release benchmarks.

| Input and path | Female TTFA / end-to-end | Male TTFA / end-to-end | Female / male RTF |
|---|---:|---:|---:|
| Short, stream vocoder | 111 / 313 ms | 107 / 273 ms | 0.177 / 0.165 |
| Long, whole utterance | 364 / 364 ms | 347 / 347 ms | 0.065 / 0.064 |
| Long, stream vocoder | 307 / 419 ms | 263 / 373 ms | 0.075 / 0.069 |
| Long, stream acoustic and vocoder | 157 / 572 ms | 152 / 545 ms | 0.102 / 0.101 |
| Three sentences, stream separately | 183 / 744 ms | 212 / 780 ms | 0.085 / 0.094 |
| Three sentences, group as paragraph and stream | 376 / 560 ms | 357 / 531 ms | 0.064 / 0.066 |

TTFA is time from the text call to the first returned 24 kHz PCM audio; the
streamed first piece was approximately 0.34–0.41 seconds of audio. End-to-end
time ends when all PCM has been returned. RTF is end-to-end time divided by
audio duration, so less than 1 is faster than real time. The measured minimum
playback buffer was positive in every streamed condition (roughly 0.11–0.28 s),
meaning no modeled playback stall for these requests. Model load took 2.26 s
separately. Network transport, encoding and playback are excluded. Raw timings
and example WAVs remain on the VM at
`runs/cpu_benchmark_preprosody/`.

The first benchmark attempt found that chunked acoustic streaming passed
Malayalam text through the Turkish frontend. Commit `b67dd74` fixed the
language argument; a focused remote Malayalam chunked-streaming check and the
full CPU benchmark then passed.

### Prosody completion and GPU generation speed

The 20k-step ProsodyNet run completed with exit code 0 and exported
`runs/prosody_ml/prosody_ema.pt`. At `T=1`, held-out dev metrics changed from
2k to 20k as follows: pitch jitter ratio 1.513 to 0.977 (recordings = 1),
pitch standard-deviation ratio 0.763 to 0.865, duration standard-deviation
ratio 0.929 to 1.024, pitch CRPS 0.267 to 0.248, duration CRPS 0.318 to
0.299, pitch distribution distance 0.097 to 0.058, and duration distribution
distance 0.077 to 0.017. Pitch correlation changed from 0.428 to 0.406:
distribution matching improved, but the model did not improve its correlation
with the particular recorded pitch contour. At `T=0.7`, final pitch correlation
was 0.444 and pitch jitter ratio 0.877. Training loss near 20k was 2.545,
with main loss 1.220; the composite loss alone is not a quality measure.

On the now-idle RTX 3060 Laptop GPU (12 GB), the same dev prompts and five
warm repetitions per condition were run with the 150k acoustic and 110k Vocos
checkpoints, first without prosody and then with the 20k prosody checkpoint.
All values below are medians in milliseconds for the female/male voices.

| Input and path | GPU without prosody TTFA / end-to-end | GPU with prosody TTFA / end-to-end | With prosody RTF, female / male |
|---|---:|---:|---:|
| Short, stream vocoder | 47 / 97; 46 / 97 | 73 / 154; 74 / 154 | 0.068 / 0.067 |
| Long, whole utterance | 49 / 49; 50 / 50 | 79 / 79; 80 / 80 | 0.013 / 0.015 |
| Long, stream vocoder | 46 / 54; 47 / 54 | 74 / 90; 73 / 85 | 0.015 / 0.016 |
| Long, stream acoustic and vocoder | 49 / 119; 46 / 110 | 76 / 158; 73 / 147 | 0.026 / 0.028 |
| Three sentences, stream separately | 47 / 152; 47 / 155 | 73 / 237; 76 / 240 | 0.027 / 0.027 |
| Three sentences, group as paragraph and stream | 47 / 61; 46 / 57 | 73 / 92; 78 / 95 | 0.011 / 0.011 |

Here each semicolon separates female and male results. TTFA and end-to-end
use the same definitions as the CPU benchmark above. Model loading (2.65 s
without prosody, 2.83 s with it), network, encoding and playback are excluded.
This is single-request throughput on one GPU, with no concurrency or serving
stack. Raw timings and example WAVs remain on the VM at
`runs/gpu_benchmark_preprosody/` and `runs/gpu_benchmark_prosody20k/`.
These latency measurements do not establish audible prosody quality; paired
listening and held-out intelligibility checks are still needed.

For simultaneous-request latency, playback stalls and the 12 GB GPU's
memory limit, see [Malayalam GPU concurrency](MALAYALAM_CONCURRENCY.md).

### Quality follow-up: slurred acoustic synthesis

In a later same-sentence listening check, the user found the original recording
and its reconstruction through the selected 110k Vocos checkpoint clear, while
unbatched FP32 synthesis was slightly slurred both with and without sampled
prosody. See the [matched A/B clips](benchmarks/malayalam-slurring-diagnostic/README.md).
On those sentences, this localizes the quality problem upstream of Vocos and rules out
`Serving.fast()`, FP16, batching, and sampled prosody as necessary causes.

With prosody disabled and Vocos fixed, [30k, 70k, 110k, and 150k acoustic
checkpoints](benchmarks/malayalam-acoustic-checkpoints/README.md) sounded
approximately alike to the user and none was satisfactory. This is informal
listening on two sentences, not a dev-set conclusion. A bounded
[guidance/temperature comparison](benchmarks/malayalam-inference-settings/README.md)
is available for listening. The two tested `norm_text` strings were unchanged
by a second Malayalam normalization pass; that check does not validate the
character-level tokenization or acoustic alignment. Prioritize acoustic text,
alignment, and data-quality diagnosis before another vocoder or prosody run.
