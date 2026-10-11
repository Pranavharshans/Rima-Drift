# ARDrift: our own autoregressive model with a one-step drifting head (#42)

Kyutai's naturalness comes from an autoregressive backbone over time with a small one-step head (Pocket TTS; the
head can be trained with drifting). Issue #42 asked whether that recipe works in Turkish. The owner's decision
(2026-10-11): no fine-tune of Kyutai's weights; build the recipe in this repository and train it from scratch on
our data. This page is that model.

- [Design](#design)
- [Training](#training)
- [Results](#results)
- [Commands](#commands)

## Design

```
text ─► text encoder ─► durations ─► frame-aligned condition [mu; h] ─┐
                                                                      ▼
   previous patch (4 mel frames) ──► causal transformer ──► state s_t ──► drift head(s_t, noise) ──► next patch
          ▲                                                                                             │
          └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| part | what | size |
|---|---|---|
| text encoder | the DiT recipe's (characters, Gaussian prior `mu`, monotonic alignment, duration predictor), trained jointly | 7.2 M |
| backbone | causal transformer over **patches of 4 mel frames** (43 ms, 23.4 steps/s), 12 layers × 512, RoPE, QK-norm, SwiGLU. Input of step t: the previous patch, the aligned condition of steps t, t+1, t+2, the speaker | 40.4 M |
| head | adaLN residual MLP (6 × 512): noise (400-d) → the next 400-d patch, a residual over the aligned prior mean, **one pass** | 18.0 M |
| vocoder | Vocos v2 (this repository's) | 13.5 M |

- **No token pitch.** The DiT is told the pitch of every character; ARDrift is not. Its intonation comes from the
  backbone, which sees the speech it already produced.
- **Aligned text instead of a text prefix.** Pocket TTS puts the text before the audio and learns the alignment by
  attention, which needs far more data than the 82 h here. Here the text encoder's alignment gives every step its
  text, and the duration predictor sets the length at inference, as in the DiT.
- **The drifting head** is trained as in Kyutai's code (`kyutai_drift_loss`): for each step (row), G = 32 head samples
  from independent noise, one positive (the recorded patch), per-row distance normalisation, the product-form field,
  one learned temperature. Kernel features: the whole patch, each frame, and each frame's harmonic detail (mel minus
  its frequency envelope).
- **Teacher forcing with input noise.** The previous patches are the recorded ones plus Gaussian noise of a random
  level up to 0.3 (normalised units) per utterance, so the backbone tolerates its own, imperfect patches.
- **Guidance** (`drift.cond_dropout`, `cfg_scale`): 15% of the fine-tune's utterances hide their text from the
  backbone; at inference the head gets `s_null + α (s - s_null)` (Pocket TTS's latent CFG, on the backbone state).
- **Fast generation** (`ARGenerator`, `Synthesizer(fast=True)`): preallocated KV caches of 512 steps and one CUDA
  graph per step (backbone + head), replayed step by step; the same output as the eager loop.

## Training

1. **Pretraining from scratch on Vyvo/tr-dataset-12** (`configs/ar_tts.yaml`): the `train` split with a light
   quality filter (DNSMOS OVRL ≥ 2.5, 8–20 letters/s, Whisper CER ≤ 0.15) and ≤ 16 s: 37,925 utterances, 81.8 h,
   722 speakers. Whole utterances (no crops), batches of 24,000 frames, 2,048 random steps per batch for the head
   (× 32 samples), AdamW 2e-4 constant (0.9, 0.999), no weight decay, clip 3, EMA 0.999. 7.7 it/s, 17 GB.
2. **Fine-tune with the studio voice** (`configs/ar_tts_ft.yaml`): the same corpus plus the studio voice (722) and
   the DiT recipe's filters (`data/tr12_eleven`), from the pretraining's 60k EMA weights, speaker table grown by one
   row, guidance dropout 0.15, 40k steps.

## Results

All rows: Vocos v2, T 0.5 for ARDrift (temperature barely matters, below), α 2 / T 0.3 for v3.2 (its operating
point); Whisper large-v3 on 8 kHz band-matched audio; UTMOSv2.

**Pretraining on tr-dataset-12** (male voice 389, which is in tr-dataset-12; Freya-100; v3.2: 1.10% / 2.915):

| step | WER, T 0.5 | UTMOSv2 | WER, T 0.8 | UTMOSv2 |
|---|---|---|---|---|
| 10k | 8.01% | 2.641 | 14.38% | 2.513 |
| 25k | 2.09% | 2.759 | 3.84% | 2.728 |
| 50k | 2.09% | 2.782 | 3.29% | 2.750 |

The learned temperature anneals to 0.060 (Kyutai: ≈ 0.056) and the data's kernel mass rises to 0.17. Speech is
intelligible by 25k steps; the DiT recipe's mel pilot was at 18.77% WER after 10k (docs/LATENTS.md).

**Fine-tune with the studio voice** (Freya-100, studio voice; v3.2: 0.44% / 2.998):

| step | cfg 1 | cfg 1.5 | cfg 2 |
|---|---|---|---|
| 10k | 0.88% / 3.009 | 0.77% / 2.931 | 5.05% / 2.609 |
| 20k | 0.55% / 3.020 | 0.66% / 2.953 | 3.51% / 2.707 |
| 30k | 0.55% / 3.024 | 0.66% / 2.956 | – |
| 40k | 0.55% / 3.026 | 0.66% / 2.936 | 2.09% / 2.724 |

Guidance does not help (male / female at cfg 2: WER 17–19%); the released setting is cfg 1. The noise temperature
barely matters either (40k, Freya-100 studio: T 0.3 / 0.5 / 0.7 / 1.0 → 0.55% WER at UTMOSv2 3.008 / 3.026 / 3.005
/ 3.020): the head's noise shapes the fine detail, the backbone the rest.

**Freya-495, the three voices** (the release protocol; ARDrift 40k against v3.2):

| voice | ARDrift WER / UTMOSv2 | v3.2 | v3.1 + BigVGAN-v2-ft |
|---|---|---|---|
| studio | **0.84%** / 3.006 | 1.33% / 3.021 | 1.23% / 2.935 |
| male | **1.79%** / 2.899 | 2.28% / 2.896 | 1.74% / 2.814 |
| female | **3.53%** / 2.665 | 3.99% / 2.722 | 3.02% / 2.752 |

Paired against v3.2 on the same 495 sentences (bootstrap 95% intervals):

| voice | word errors | UTMOSv2 | DNSMOS OVRL | speaker sim. | F0 std |
|---|---|---|---|---|---|
| studio | **−19** [−32, −7] | −0.015 [−0.038, +0.009] | −0.000 | **+0.009** | −0.20 st |
| male | −19 [−44, +5] | +0.003 [−0.026, +0.031] | −0.013 | **+0.019** | −0.15 st |
| female | −18 [−50, +14] | **−0.058** [−0.087, −0.028] | +0.016 | **+0.045** | −0.72 st |

**Against the recordings** (the 100 studio `val` utterances, the same texts; `drifting-tts prosody`):

| system | DTW F0 r | F0 RMSE | F0 std (rec. 3.68) | pauses / utt (1.39) | CER* | UTMOSv2 (rec. 3.093) | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3.2, sentence by sentence | 0.579 | 3.25 | 3.53 | 1.54 | 0.41% | 2.990 | 0.940 |
| v3.2, paragraph mode | 0.577 | 3.25 | 3.50 | 1.15 | 0.38% | 2.977 | 0.939 |
| ARDrift, sentence by sentence | 0.607 | 3.13 | 3.51 | 1.60 | 0.53% | 3.029 | 0.945 |
| **ARDrift, one pass** | **0.633** | **2.98** | 3.34 | 0.59 | 0.42% | **3.066** | 0.944 |

(CER* leaves out the one utterance whose text repeats a sentence, which Whisper drops; docs/PROSODY.md.)

- **The closest intonation measured in this project.** With no pitch given, ARDrift's contour matches the
  recordings better than v3.2's sampled token pitch (DTW F0 r 0.633 against 0.579; earlier systems on this set
  reached 0.577–0.617, v3.1's regressors in one pass the highest, docs/PROSODY_MODEL.md), and its UTMOSv2 is the
  closest to the recordings'.
- **More intelligible and closer to the voice.** Fewer word errors on Freya-495 for every voice (studio
  significantly), higher WavLM-ECAPA similarity to each voice's recordings.
- **Narrower pitch spread than v3.2's sampler,** most for the female voice (2.32 against 3.04 st on Freya-495),
  whose UTMOSv2 is also 0.06 lower; one pass leaves fewer internal pauses than the recordings (0.59 against 1.39).
- **Slower, but streaming from the first step.** On an idle RTX 5090 with CUDA graphs (`fast=True`, streaming,
  Vocos v2): first audio after 36–37 ms for any text length (v3.2: 6–8 ms), RTF 0.048 (v3.2: 0.001–0.005). A
  step (backbone + head, batch 1) takes ~2 ms; the first audio needs the 16 steps of the first vocoder window
  and its context.


## Commands

```bash
drifting-tts train-ar --config configs/ar_tts.yaml --workdir runs/ar_pre                 # tr-dataset-12, from scratch
drifting-tts train-ar --config configs/ar_tts_ft.yaml --workdir runs/ar_ft \
    train.init_from=runs/ar_pre/model_ema_60000.pt                                    # + the studio voice
drifting-tts calibrate-durations --model runs/ar_ft/model_ema.pt --speakers 722 389 323
drifting-tts benchmark --model runs/ar_ft/model_ema.pt --vocoder vocos-v2 --num 100 --speaker studio \
    --temperature 0.5 --cfg 1.5
```

```python
tts = Synthesizer("runs/ar_ft/model_ema.pt", "cuda", vocoder="vocos-v2", fast=True)
wav, _ = tts("Merhaba, nasılsınız?", speaker="studio", temperature=0.5, cfg_scale=1.5)
```
