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

<!-- ar-results -->

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
