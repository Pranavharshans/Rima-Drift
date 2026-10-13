# Malayalam v3.2 training TODO

Status: **planning**. Target: a Malayalam, two-voice, v3.2-style TTS pipeline trained with
`ai4bharat/Rasa` Malayalam on a separate GPU machine. No Malayalam checkpoint is trained or validated yet.

**Two passes, with a hard boundary:** first implement and push *all* stages' code,
configurations, run commands, validation checks and checkpoint integration. No dataset
download, dataset processing, scoring or GPU job runs on the local computer. Do not start
**any GPU job** until all implementation checkboxes below are complete and the complete
pipeline is pushed to `origin`. Then, on the separate machine, fetch and process Rasa,
run the pilots and training jobs in dependency order, and review each acceptance gate.
Push small evidence and any fixes after each remote stage. Record the commit, dataset
revision, machine/GPU and result below. Keep gated audio, credentials and large checkpoints
out of Git; retain their locations, hashes and provenance in the stage record. Do not
advance a failed stage merely because its scheduled training steps finished.

The initial dataset schema is gated. The remote-only ingestion code must inspect it and
fail clearly on unexpected fields before full processing. If the first remote inspection
reveals a mismatch, fix and push the code before proceeding. This is a validation/fix
gate, not authorization to begin training early.

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
independently once their dependencies exist. Every implementation stage gets its own push;
remote results and fixes get follow-up pushes.

## Implementation pass: complete and push before any GPU job

The remote ingestion command is `drifting-tts rasa-prepare --revision <RASA_COMMIT_SHA>
--out <REMOTE_GATED_DIR>`. Use `--inspect-only` first. It downloads only
`Malayalam/train-*.parquet` and `Malayalam/test-*.parquet` to the remote Hugging Face
cache, then writes canonical `train.parquet`, `dev.parquet`, `val.parquet` and `audit.json`.
The public Rasa card specifies these fields and source splits, but actual parquet contents
still require a gated remote inspection. The `val` split comes from Rasa's original test
split; `dev` is a deterministic subset of its original train split. The canonical parquet
still contains gated audio and must stay on the remote machine. The downstream feature
preparation command and Malayalam text frontend are implemented in stage 1.
Install the repository with `pip install -e '.[rasa,score,eval,bigvgan]'` on the remote
machine; authenticate to the gated dataset there before running ingestion.

After the gated remote inspection, prepare the canonical parquet files with
`drifting-tts prepare --parquet-glob '<REMOTE_GATED_DIR>/*.parquet' --out
<REMOTE_PREPARED_DIR> --language ml --preserve-splits --backend bigvgan --f0
--save-audio`. This creates 24 kHz waveforms/mels and a `rejections.jsonl` audit. The
Malayalam frontend preserves Unicode script and rejects unsupported symbols or digits
that need a reviewed spoken form; it does not silently delete them. Review the rejection
counts and examples on the remote machine. Run `drifting-tts score --data
<REMOTE_PREPARED_DIR>` there; ASR uses Malayalam. The Turkish rate and CER thresholds
are disabled in `configs/tts_v3_ml.yaml` until measured Malayalam thresholds are approved.
The Rasa `val` split is reserved for final reporting; use `dev` for tuning. The evaluation
result includes metrics by speaker. Full run commands and final thresholds belong to
the later implementation stages.

- [x] **0 — Data code:** remote-only Rasa Malayalam fetch with pinned revision and schema
      inspection; stable speaker mapping; deterministic splits; duplicate, duration and
      missing-field audits; source audio to 24 kHz/mel preparation; audit reports.
- [x] **1 — Text/evaluation code:** Malayalam normalization and vocabulary, checkpoint
      compatibility checks, quality scoring, held-out manifests and per-voice evaluation.
- [x] **2 — Mel-MAE code:** Malayalam config, pilot/full-run commands, reconstruction and
      checkpoint checks.
- [x] **3 — Acoustic code:** two-speaker config, alignment/overfit pilot, full-run command,
      duration calibration and per-voice checkpoint selection.
- [x] **4 — Vocos first-run code:** 40k-step config, pretrained source, two-voice development copy-synthesis checks.
- [x] **5 — Vocos continuation code:** 160k additional steps, source checkpoint and snapshot selection checks.
- [x] **6 — Prosody code:** frozen-acoustic target cache, pilot/full-run config, temperature
      calibration and deterministic fallback checks.
- [x] **7 — Integration code:** Malayalam checkpoint loading, two-voice synthesis, held-out
      inference/listening protocol, provenance and release artifact checks.
- [x] **Implementation gate:** review and push the complete remote runbook and all code;
      record the final implementation commit after the stage-7 push before any GPU job is submitted.

Implementation stages 0–7 are code-complete and pushed. Their gated-data behavior and
Malayalam speech quality are **unverified** until remote execution. Number-bearing and
unsupported transcripts are rejected for review; a Malayalam number verbalizer can be
added after the remote rejection audit shows which forms are actually present.

### Stage 2 remote runbook (do not run until the implementation gate)

Use the prepared `data/malayalam` directory created from the canonical Rasa parquet.
The `bigvgan` mel backend must match the acoustic and Vocos stages. Keep pilot and full
checkpoints in distinct directories; `train-mae` resumes automatically from `last.pt`
within the selected workdir.

```bash
drifting-tts train-mae --config configs/mae2d_ml.yaml --workdir runs/mae2d_ml_pilot \
  train.steps=200 train.batch_size=8 train.warmup=20 train.cls_finetune_steps=0 \
  train.log_every=20 train.save_every=100
drifting-tts check-mae --config configs/mae2d_ml.yaml \
  --checkpoint runs/mae2d_ml_pilot/mae_ema.pt --split dev --per-speaker 4 \
  --expected-step 200 --out runs/mae2d_ml_pilot/check

drifting-tts train-mae --config configs/mae2d_ml.yaml --workdir runs/mae2d_ml
drifting-tts check-mae --config configs/mae2d_ml.yaml \
  --checkpoint runs/mae2d_ml/mae_ema.pt --split dev --per-speaker 4 \
  --expected-step 60000 --out runs/mae2d_ml/check
```

Pilot gate: both voices have dev examples; training loss and masked reconstruction are
finite, and loss trends down rather than diverging. Inspect `report.json` and the masked
input/reconstruction arrays in `reconstruction_samples.npz` on the remote machine.
The zero-output MSE is a reference, not an automatic pass threshold. Full-run gate:
the 60k EMA checkpoint loads, matches Malayalam/BigVGAN/100-mel metadata, has finite
activations for the acoustic drift loss, and its per-voice reconstruction is reviewed.
Record the actual GPU-hours, selected checkpoint hash, dev results and any failure.

### Stage 3 remote runbook (do not run until the implementation gate)

The acoustic config is `configs/tts_v3_ml.yaml`: two fixed speaker IDs, Malayalam text,
the stage-2 2-D Mel-MAE checkpoint, pitch conditioning, BigVGAN mels and 150k steps.
The stage-1 Turkish speaking-rate and CER cutoffs remain disabled until Malayalam
distributions are reviewed. Pilot training selects 16 usable train utterances per voice;
it never uses `dev` or `val` as training data. The pilot has a separate workdir and
reduces batch size, generator samples and steps to bound GPU use.

```bash
drifting-tts train --config configs/tts_v3_ml.yaml --workdir runs/tts_v3_ml_pilot \
  train.pilot_per_speaker=16 train.steps=1000 train.batch_size=2 train.batch_frames=3000 \
  train.warmup=100 train.num_workers=2 train.log_every=20 train.save_every=200 \
  train.sample_every=0 train.keep_snapshots=false train.compile=false \
  drift.gen_per_cond=4 drift.uncond_per_cond=4 drift.uncond_bank=128
drifting-tts calibrate-durations --model runs/tts_v3_ml_pilot/model_ema.pt --num 16 --dry-run
drifting-tts evaluate --model runs/tts_v3_ml_pilot/model_ema.pt --split dev \
  --num-per-speaker 8 --temperature 0.5 --cfg 2 --sv none --mos none \
  --out runs/tts_v3_ml_pilot/eval_dev

drifting-tts train --config configs/tts_v3_ml.yaml --workdir runs/tts_v3_ml
```

Pilot gate: `loss`, `prior`, `duration`, `pitch`, `unaligned_tokens` and
`alignment_max_share` remain finite; inspect the alignment metrics and loss trend,
the per-speaker calibration ratios, and generated audio. A 1k-step pilot is an
integration and overfit check, not a release-quality WER target. If it fails, preserve
its logs and fix the code/data before the full run. The full workdir keeps EMA
snapshots every 5k steps and resumes from `last.pt` in the same directory.

Calibrate *each candidate* on training utterances, then compare the same balanced dev
sentences with the same evaluation settings. For example:

```bash
for step in 100000 125000 150000; do
  model="runs/tts_v3_ml/model_ema_${step}.pt"
  drifting-tts calibrate-durations --model "$model" --num 200
  drifting-tts evaluate --model "$model" --split dev --num-per-speaker 100 \
    --temperature 0.5 --cfg 2 --sv none --mos none \
    --out "runs/tts_v3_ml/eval_dev_${step}"
done
drifting-tts select-acoustic --results runs/tts_v3_ml/eval_dev_*/results.json \
  --out runs/tts_v3_ml/checkpoint_ranking.json
```

The ranking requires both voices and identical dev settings. It ranks by the worse
voice's WER, then mean WER; it is a review aid, not automatic approval. Listen to both
voices, inspect per-speaker CER/WER and alignment, then record the selected checkpoint
and SHA-256. Keep `val` untouched for final reporting in stage 7. A later Vocos run
may change absolute quality, so the selected acoustic checkpoint remains provisional
until the complete pipeline passes its held-out gate.

### Stage 4 remote runbook (do not run until the implementation gate)

Use the **selected and calibrated** stage-3 acoustic EMA checkpoint. The Malayalam
config inherits the upstream 40k-step Vocos GAN recipe, which initializes the
generator from `charactr/vocos-mel-24khz`; the discriminators start fresh. Both
acoustic and prepared-data mels must be `bigvgan`. The trainer checks checkpoint
language, speaker count/map, and prepared-data backend before training. Keep the
pilot separate; the full workdir resumes from its own `last.pt` if interrupted.

```bash
ACOUSTIC_CKPT=runs/tts_v3_ml/model_ema_SELECTED_STEP.pt
sha256sum "$ACOUSTIC_CKPT"
drifting-tts finetune-vocoder --config configs/vocoder_vocos_ml.yaml \
  --workdir runs/vocos_ml_pilot "tts.path=$ACOUSTIC_CKPT" \
  train.steps=200 train.disc_warmup_steps=20 train.sample_every=100 \
  train.save_every=100 train.log_every=20
drifting-tts finetune-vocoder --config configs/vocoder_vocos_ml.yaml \
  --workdir runs/vocos_ml "tts.path=$ACOUSTIC_CKPT"
sha256sum runs/vocos_ml/last.pt runs/vocos_ml/vocos_ft.pt
tensorboard --logdir runs/vocos_ml/tb
```

The TensorBoard audio triplets `real`, `recorded_mel` (copy synthesis), and `gta`
(acoustic generated mel) come from the `dev` split and include one female and one
male voice. At the pilot and each 5k-step checkpoint, compare each voice's triplet
for clipping, buzz, dropped sounds, pitch change and intelligibility; inspect
generator/discriminator and `mel_gta`/`mel_recorded` loss trends. Stop and diagnose
if the pilot has nonfinite loss, mismatched framing or a missing voice. At 40k,
retain the full `last.pt` with optimizer and discriminator state for stage 5,
the exported `vocos_ft.pt`, snapshots and hashes. Record the selected acoustic
path/hash, pretrained Vocos revision, prepared dataset revision, GPU-hours, and
per-voice listening notes. Do not use Rasa `val` for tuning.

### Stage 5 remote runbook (do not run until the implementation gate)

The Malayalam continuation inherits the upstream Vocos v2 loss rebalance and
cosine schedule. It starts a new 160k-step run from the complete stage-4 40k-step
`last.pt`, loading generator, discriminators and optimizer states. The source
checkpoint gate rejects wrong step counts or incomplete state. Keep the same
selected acoustic checkpoint and prepared dataset as stage 4.

```bash
ACOUSTIC_CKPT=runs/tts_v3_ml/model_ema_SELECTED_STEP.pt
sha256sum runs/vocos_ml/last.pt
drifting-tts finetune-vocoder --config configs/vocoder_vocos_v2_ml.yaml \
  --workdir runs/vocos_v2_ml "tts.path=$ACOUSTIC_CKPT"

for step in 120000 140000 160000; do
  vocoder="runs/vocos_v2_ml/vocos_ft_${step}.pt"
  drifting-tts evaluate --model "$ACOUSTIC_CKPT" --vocoder "$vocoder" \
    --split dev --num-per-speaker 100 --temperature 0.5 --cfg 2 \
    --sv none --mos none --out "runs/vocos_v2_ml/eval_dev_${step}"
done
drifting-tts select-vocoder --results runs/vocos_v2_ml/eval_dev_*/results.json \
  --out runs/vocos_v2_ml/checkpoint_ranking.json
```

These are example candidate steps in the **continuation** workdir; include any
earlier or later snapshots with stronger listening quality. The ranking compares
identical dev utterances for both voices, checks both generated and copy-synthesis
WER, and refuses a missing recording reference. Listen to the saved audio and the
TensorBoard `real`/`recorded_mel`/`gta` triplets, especially buzz, pitch and
consonants. Select by review, record the selected export hash and the source
`last.pt` hash, and retain all snapshot files. Never select on Rasa `val`.

### Stage 6 remote runbook (do not run until the implementation gate)

Cache MAS duration and token-pitch targets from the **selected** acoustic checkpoint,
using train and dev only. The Malayalam prosody config rejects a cache built from a
different acoustic path, a cache containing `val`, a non-Malayalam acoustic model,
or a speaker-map mismatch. It calibrates both voice IDs from training rows. Keep
the pilot and full runs separate; full EMA snapshots are retained every 2k steps.

```bash
ACOUSTIC_CKPT=runs/tts_v3_ml/model_ema_SELECTED_STEP.pt
VOCODER_CKPT=runs/vocos_v2_ml/vocos_ft_SELECTED_STEP.pt
drifting-tts prosody-cache --model "$ACOUSTIC_CKPT" \
  --data data/malayalam --splits train dev --out runs/prosody_ml_cache/targets.pt
drifting-tts train-prosody --config configs/prosody_drift_ml.yaml \
  --workdir runs/prosody_ml_pilot "tts=$ACOUSTIC_CKPT" \
  train.steps=200 train.warmup=20 train.eval_every=100 train.save_every=100
drifting-tts train-prosody --config configs/prosody_drift_ml.yaml \
  --workdir runs/prosody_ml "tts=$ACOUSTIC_CKPT"

for voice in female male; do
  drifting-tts prosody --model "$ACOUSTIC_CKPT" --vocoder "$VOCODER_CKPT" \
    --release-prosody runs/prosody_ml/prosody_ema.pt \
    --systems recording copy predicted release --split dev --speaker "$voice" \
    --num 100 --diversity 4 --diversity-systems release \
    --out "runs/prosody_ml/eval_dev_${voice}"
done
```

Inspect pilot loss, `dev_history.json`, sampled rhythm and pitch, ASR, voice
identity and both voices' listening samples. The prosody evaluation routes
Malayalam text and ASR language correctly. Try candidate temperatures and
`--release-prosody-durations regressor` on the **dev** split if sampled duration
hurts either voice. Use deterministic prosody (`--prosody none` at synthesis)
as the fallback if sampling degrades either voice. Record the selected prosody
checkpoint hash and the cache/acoustic hashes. Do not tune on `val`.

### Stage 7 remote runbook (only after the implementation gate and stages 0–6 pass)

Use the exact selected and calibrated checkpoints from the earlier gates. The
release check loads them on CPU, checks Malayalam vocabulary and speaker mapping,
BigVGAN mel framing, Vocos metadata, the prosody acoustic fingerprint and both
duration calibrations, then writes SHA-256 provenance. It does not certify speech
quality. Run it first without held-out reports; keep its JSON beside the model
artifacts and preserve the dataset revision and implementation commit.

```bash
ACOUSTIC_CKPT=runs/tts_v3_ml/model_ema_SELECTED_STEP.pt
VOCODER_CKPT=runs/vocos_v2_ml/vocos_ft_SELECTED_STEP.pt
PROSODY_CKPT=runs/prosody_ml/prosody_ema.pt
RASA_REVISION=PINNED_RASA_COMMIT_SHA
IMPLEMENTATION_COMMIT=$(git rev-parse HEAD)
drifting-tts check-ml-release --acoustic "$ACOUSTIC_CKPT" \
  --vocoder "$VOCODER_CKPT" --prosody "$PROSODY_CKPT" \
  --dataset-revision "$RASA_REVISION" --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --out runs/release_ml/manifest.json

for voice in female male; do
  drifting-tts synthesize --model "$ACOUSTIC_CKPT" --vocoder "$VOCODER_CKPT" \
    --prosody "$PROSODY_CKPT" --speaker "$voice" \
    --text 'നമസ്കാരം. ഇന്ന് സുഖമാണോ?' --pause 0.15 \
    --out "runs/release_ml/${voice}_sentences.wav"
  drifting-tts synthesize --model "$ACOUSTIC_CKPT" --vocoder "$VOCODER_CKPT" \
    --prosody "$PROSODY_CKPT" --speaker "$voice" --paragraph \
    --text 'നമസ്കാരം. ഇന്ന് സുഖമാണോ?' --pause 0.15 \
    --out "runs/release_ml/${voice}_paragraph.wav"
  drifting-tts prosody --model "$ACOUSTIC_CKPT" --vocoder "$VOCODER_CKPT" \
    --release-prosody "$PROSODY_CKPT" \
    --systems recording copy predicted release release-para \
    --split val --speaker "$voice" --num 100 \
    --out "runs/release_ml/eval_val_${voice}"
done

drifting-tts check-ml-release --acoustic "$ACOUSTIC_CKPT" \
  --vocoder "$VOCODER_CKPT" --prosody "$PROSODY_CKPT" \
  --dataset-revision "$RASA_REVISION" --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --heldout-female runs/release_ml/eval_val_female/results.json \
  --heldout-male runs/release_ml/eval_val_male/results.json \
  --out runs/release_ml/manifest.json
```

The final check requires valid Malayalam ASR CER/WER for both voices and verifies
that both held-out reports used these exact checkpoint paths. Review the WAVs,
per-voice intelligibility, identity, copy quality, sampled versus deterministic
rhythm/pitch, clipping, and paragraph pauses. Prefer fixed `--pause 0.15` until
a Malayalam punctuation pause policy is calibrated; the upstream policy is
Turkish-specific. If sampled prosody fails a voice, use deterministic prosody
as the documented fallback and evaluate that exact mode before release. Keep
Rasa audio, generated speech and large checkpoints on the remote machine;
commit only manifests, concise metrics and approved documentation updates.

## Remote execution pass: only after the implementation gate

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
