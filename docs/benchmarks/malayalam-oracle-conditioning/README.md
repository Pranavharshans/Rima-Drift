# Malayalam acoustic conditioning diagnosis

The acoustic TTS model was trained with MAS timing and token pitch derived from each recording. Free-text synthesis predicts both. The earlier Vocos GTA previews also used reference timing and pitch. This experiment tests that train/inference gap without changing the acoustic checkpoint, speaker, sentence, CFG (2), noise temperature (0.5), style seed (42), or stock BigVGAN-v2 vocoder.

The two dev recordings are diagnostic references. Their reference timing is the checkpoint's MAS alignment and their reference pitch is derived from F0. These oracle inputs are **not available for new text at deployment**.

| Voice | Recording | Predicted duration + pitch | Reference duration only | Reference pitch only | Reference duration + pitch |
|---|---|---|---|---|---|
| Female | [WAV](female_recording.wav) | [WAV](female_predicted_duration_predicted_pitch.wav) | [WAV](female_reference_duration_predicted_pitch.wav) | [WAV](female_predicted_duration_reference_pitch.wav) | [WAV](female_reference_duration_reference_pitch.wav) |
| Male | [WAV](male_recording.wav) | [WAV](male_predicted_duration_predicted_pitch.wav) | [WAV](male_reference_duration_predicted_pitch.wav) | [WAV](male_predicted_duration_reference_pitch.wav) | [WAV](male_reference_duration_reference_pitch.wav) |

For the female sentence, predicted timing totaled 676 mel frames and reference timing 759 (7.21 versus 8.10 seconds of generated audio). For the male sentence, the totals were 661 and 687 (7.05 versus 7.33 seconds). Total duration does not measure syllable clarity; listen to matched words and consonants. Within each duration row, the random mel input and style seed are held fixed, isolating pitch. Between duration rows the generated mel lengths differ, so the noise shapes necessarily differ.

If reference duration alone improves clarity, investigate the duration predictor or its calibration. If reference pitch alone improves clarity, investigate the pitch predictor and teacher-forcing gap. If both references remain slurred, inspect the generator, alignment, and long-sequence inference. Do not launch retraining based on the duration totals alone.

The exact source IDs and output lengths are in [manifest.json](manifest.json). Reproduce on a CUDA machine with the prepared Rasa Malayalam dataset:

```bash
python -m scripts.diagnose_malayalam_oracle_conditioning \
  --model runs/tts_v3_ml/model_ema_150000.pt \
  --data data/malayalam --out runs/oracle_conditioning_ml
```

Source recordings: ai4bharat/Rasa Malayalam, CC BY 4.0.
