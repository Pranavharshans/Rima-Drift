# Malayalam v3.2 training TODO

Status: **planning**. Target: a Malayalam, two-voice, v3.2-style TTS pipeline trained with
`ai4bharat/Rasa` Malayalam on a separate GPU machine. No Malayalam checkpoint is trained or validated yet.

Check off a stage only after its acceptance checks pass. For each stage, commit its code,
configuration and small evidence files, then push that commit to `origin`. Record the commit,
dataset revision, machine/GPU and result below. Keep gated audio, credentials and large
checkpoints out of Git; retain their locations, hashes and provenance in the stage record.
Do not advance a failed stage merely because its scheduled training steps finished.

## Planning estimates

LOC means **estimated added or changed lines**, including configuration and focused tests;
it is not a count of work already done. GPU hours assume one 32 GB NVIDIA GPU near RTX 5090
performance. They are estimates for Malayalam, not measured runs. Preprocessing, scoring,
listening and transfers add wall-clock time. The upstream Turkish [training guide](TRAINING.md)
reports 1.5 hours for 60k Mel-MAE steps and 13.6 hours for 150k acoustic steps. Its
[v3.2 Vocos recipe](VOCODERS.md) uses 40k initial steps plus 160k continuation steps; the
[prosody recipe](PROSODY_MODEL.md) trains separately against a frozen acoustic model.

| Stage | Estimated code change | Estimated GPU training | Dependency |
|---|---:|---:|---|
| 0. Rasa ingestion, speaker map and splits | 150–300 LOC | None | Dataset access |
| 1. Malayalam text frontend and quality/evaluation | 400–800 LOC | None; scoring may use GPU | Stage 0 |
| 2. Mel-MAE | 20–60 LOC | 1–3 h | Stages 0–1 |
| 3. Acoustic TTS, including joint alignment/duration/pitch | 80–180 LOC | 14–30 h | Stages 0–2 |
| 4. Vocos initial fine-tune, 40k steps | 20–50 LOC | 3–6 h | Stage 3 |
| 5. Vocos continuation, 160k additional steps | 20–50 LOC | 10–20 h | Stage 4 |
| 6. Stochastic prosody | 80–200 LOC | 1–3 h | Stage 3 |
| 7. Release integration and held-out validation | 80–160 LOC | No fixed training time | Stages 3, 5–6 |
| **Total** | **~850–1,800 LOC** | **~30–60 h for one set of runs** | |

Reserve roughly **40–80 GPU-hours** for pilots, evaluation and a likely repeat of a weak
stage. Code and review effort is roughly **2–4 focused engineering weeks**; update these
estimates after a short benchmark on the chosen machine. Stages 5 and 6 can be evaluated
independently once their dependencies exist, but each gets its own commit and push.

## Checklist and acceptance gates

### 0. Rasa Malayalam dataset

- [ ] Confirm gated dataset access, permitted use and exact dataset revision.
- [ ] Select Malayalam only; retain source-quality audio and transcripts. Map female and male
      voices to distinct stable speaker IDs. Prepare repository-compatible 24 kHz audio/mels.
- [ ] Create fixed, speaker-aware train/development/test manifests. Check duplicates, sample
      rates, durations, missing text/audio and leakage across splits.
- [ ] Save a small manifest summary and a bounded sample audit; keep the gated content out of Git.
- [ ] Commit and push this stage. Record: commit ___; dataset revision ___; sample counts ___;
      storage location ___; issues ___.

### 1. Malayalam text and quality gates

- [ ] Implement Malayalam Unicode normalization, character inventory, punctuation and number
      handling. Verify round-trip/coverage on actual Rasa transcripts; flag unsupported text.
- [ ] Route training and inference through the same frontend. Version the vocabulary with
      the checkpoint so a Turkish vocabulary cannot silently load for Malayalam.
- [ ] Adapt or replace Turkish ASR agreement, speaking-rate filters and evaluation with
      Malayalam-capable checks. Define an immutable held-out sentence and listening set for
      both speakers; do not tune on test data.
- [ ] Run focused text/data checks and review a sample of flagged rows.
- [ ] Commit and push this stage. Record: commit ___; text coverage ___; rejected/reviewed rows
      ___; evaluation protocol ___.

### 2. Mel-MAE

- [ ] Add Malayalam paths/configuration and run a short pilot to verify batches, loss and
      reconstruction before the full 60k-step target.
- [ ] Train, inspect reconstruction samples and record checkpoint hash and GPU-hours.
- [ ] Commit and push configuration, command and concise results. Record: commit ___;
      checkpoint/hash ___; GPU-hours ___; decision ___.

### 3. Acoustic TTS

- [ ] Configure full-weight acoustic training using the Malayalam frontend, both speaker IDs
      and Mel-MAE output. The text encoder, alignment, duration, pitch and mel generator are
      trained together, rather than as separate jobs.
- [ ] Run a bounded alignment/overfit pilot; inspect synthesized audio and per-speaker losses.
- [ ] Run the full training target only after the pilot passes. Calibrate duration and select
      a checkpoint on development data; evaluate intelligibility for each voice separately.
- [ ] Commit and push configuration, command and evidence. Record: commit ___;
      checkpoint/hash ___; GPU-hours ___; per-voice metrics ___; decision ___.

### 4. Vocos initial fine-tune

- [ ] Confirm the acoustic model's mel format matches the Vocos recipe. Fine-tune from the
      documented pretrained Vocos weights for the initial 40k-step run.
- [ ] Listen to copy synthesis and generated mels for both voices; check artifacts and pitch.
- [ ] Commit and push configuration and results. Record: commit ___; checkpoint/hash ___;
      GPU-hours ___; decision ___.

### 5. Vocos continuation

- [ ] Resume the completed stage-4 checkpoint, including optimizer/discriminator state, for
      **160k more steps** (200k cumulative); keep candidate snapshots.
- [ ] Select a snapshot on development audio using intelligibility, listening quality and
      copy-synthesis checks; the final step is not automatically the winner.
- [ ] Commit and push configuration and results. Record: commit ___; selected snapshot/hash
      ___; GPU-hours ___; decision ___.

### 6. Stochastic prosody

- [ ] Freeze the selected acoustic checkpoint and cache Malayalam duration/pitch targets.
- [ ] Train the separate drift prosody predictor (20k-step config target), first with a short
      pilot. Calibrate temperature on development data for both voices.
- [ ] Compare sampled and deterministic prosody for rhythm, pitch and intelligibility. Keep
      deterministic prosody as a fallback if sampling harms either voice.
- [ ] Commit and push configuration and results. Record: commit ___; checkpoint/hash ___;
      GPU-hours ___; per-voice decision ___.

### 7. Integration and release gate

- [ ] Load the exact Malayalam acoustic, Vocos and prosody checkpoints in one inference path.
      Verify text handling, speaker selection, sentence/paragraph synthesis and output audio.
- [ ] Run held-out inference and a listening review for both voices; report intelligibility,
      artifacts, speaker identity and known limitations without borrowing Turkish metrics.
- [ ] Save reproducible commands, config/data revisions, checkpoint hashes, evaluation
      outputs and example audio references. Keep large/gated artifacts in approved storage.
- [ ] Mark the README status as trained only if these checks pass. Commit and push the release
      evidence. Record: commit ___; artifact locations/hashes ___; final decision ___.

## Scope notes

- The upstream v3.2 release reuses v3.1 Turkish acoustic weights. A **new Malayalam acoustic
  model is still required**; copying the Turkish release weights is not Malayalam training.
- Vocos is the v3.2 target vocoder. BigVGAN may serve as a comparison, but is not a required
  extra training stage. Turkish BERT/word features are not part of the initial baseline.
- All time and LOC numbers are provisional until the actual dataset, GPU and throughput are
  measured. Do not treat a training-step count alone as a quality gate.
