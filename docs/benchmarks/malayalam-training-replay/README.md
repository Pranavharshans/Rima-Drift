# Malayalam training preview replay

This comparison uses validation items 1 and 2, which were among the first four female items used by `train.py` for every training preview. The text, speaker, random seed, CFG 2, temperature 0.5, and direct acoustic `model.synthesize` call match that preview. The saved training WAV and both replay WAVs use the same acoustic checkpoint. The two replays decode the **same generated mel** with either stock BigVGAN-v2 (the training-preview vocoder) or the later fine-tuned Vocos checkpoint.

For each utterance and checkpoint, listen in this order:

1. `*_saved_training.wav` — the actual file saved during training.
2. `*_bigvgan_replay.wav` — exact prompt and settings replayed through stock BigVGAN-v2.
3. `*_vocos_replay.wav` — the same generated mel decoded through fine-tuned Vocos.
4. `utt*_recording.wav` — the source recording for context.

The saved-training and BigVGAN replay WAVs have the same sample count and waveform correlation above 0.999999 for all four pairs. Their small PCM differences are consistent with WAV quantization. This verifies that the replay reproduces the training preview. It does **not** establish which audio sounds better; that requires listening.

The newer free-text samples also changed the prompt and speaker mix, so the familiar validation items alone cannot establish quality on new sentences. A clear Vocos reconstruction of a recorded mel does not rule out a Vocos problem on **generated** mels. Comparing the two vocoders on the same generated mel isolates that question.

Reproduce on a CUDA machine with the checkpoints and prepared dataset present:

```bash
python -m scripts.replay_malayalam_training_samples \
  --models 70000:runs/tts_v3_ml/model_ema_70000.pt 150000:runs/tts_v3_ml/model_ema_150000.pt \
  --vocos runs/vocos_v2_ml/vocos_ft_110000.pt \
  --data data/malayalam --training-samples runs/tts_v3_ml/samples \
  --indices 1 2 --out runs/training_replay_ml
```

The `manifest.json` records the exact validation IDs and text. Dataset: ai4bharat/Rasa Malayalam, CC BY 4.0.
