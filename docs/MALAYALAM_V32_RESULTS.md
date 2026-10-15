# Malayalam v3.2 training results and status

Status recorded **15 October 2026**. This is a running experiment on a remote
NVIDIA RTX 3060 Laptop GPU (12 GB). Rasa data, checkpoints, logs and WAV files
remain on that machine; they are not stored in Git. The code revision used for
the current runs is `417735f`.

## Data and completed components

| Component | Result | Evidence and status |
|---|---|---|
| Rasa Malayalam preparation | 32,466 usable utterances, approximately 49.3 hours, female and male speakers | Prepared remote dataset; approximately 12 GB. |
| Mel-MAE | Completed 60,000 steps | Final `runs/mae2d_ml/mae_ema.pt`; final reported reconstruction loss about 0.042. |
| Acoustic TTS | Completed 150,000 steps in 30.92 hours | The final calibrated EMA checkpoint was selected for subsequent stages: `runs/tts_v3_ml/model_ema_150000.pt`. SHA-256: `2bfa4a59057e9564ab723ab6e047fa5c81c1c061ef44d2b12f58667db4488d76`. |
| Initial Vocos | Completed 40,000 steps | `runs/vocos_ml/last.pt` (full continuation state; SHA-256 `4bdf157e988c49ee13928130b179e6b81df03d95f97e61a213ce94f174355b73`); exported generator `runs/vocos_ml/vocos_ft_40000.pt`. |
| Vocos continuation | **Running**, step 54,600 of 160,000 additional steps at the last status check | Started from the full step-40,000 Vocos state, including generator, discriminators and optimizer state. Latest checked speed was about 1.14 steps/s with 11.1 GB GPU memory. Its step counter starts at zero; step 54,600 means 94,600 cumulative Vocos steps. |
| Stochastic prosody | Not started | Planned 20,000-step run after selecting the Vocos checkpoint. |
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

1. Keep the Vocos continuation running while matching-utterance listening
   shows gains. Compare both voices at later saved checkpoints, especially
   sample 0's jitter and sample 1's retained clarity. Choose the checkpoint
   on dev listening and valid evaluation, not training loss alone.
2. Freeze the selected acoustic and Vocos checkpoints; cache Malayalam
   duration/pitch targets, run the 200-step prosody pilot, then train the
   planned 20k-step prosody model. Compare sampled prosody with deterministic
   timing and pitch; keep the deterministic path if prosody harms either voice.
3. Run final two-voice listening and held-out checks on the exact assembled
   model. Record dataset revision, checkpoint hashes and validated metrics
   before calling the Malayalam model released.

The active continuation log is `/workspace/vocos_v2_ml_full.log`; its output
directory is `/workspace/Rima-Drift/runs/vocos_v2_ml/`. The status above is a
snapshot, not a live counter.
