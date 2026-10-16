# Malayalam acoustic inference settings

After the 30k, 70k, 110k, and 150k acoustic checkpoints all sounded similarly
slurred, this bounded sweep checks whether the standard acoustic inference
settings are responsible. It holds the 150k acoustic checkpoint, 110k Vocos,
text, source voice, seed, and **unbatched FP32 inference without prosody** fixed.
Only classifier-free guidance (`cfg`) and acoustic noise temperature change.

| Voice | CFG 1, T 0.3 | CFG 2, T 0.3 (baseline) | CFG 3, T 0.3 | CFG 2, T 0.5 |
|---|---|---|---|---|
| Female | [WAV](female_cfg1_temp0.3.wav) | [WAV](female_cfg2_temp0.3.wav) | [WAV](female_cfg3_temp0.3.wav) | [WAV](female_cfg2_temp0.5.wav) |
| Male | [WAV](male_cfg1_temp0.3.wav) | [WAV](male_cfg2_temp0.3.wav) | [WAV](male_cfg3_temp0.3.wav) | [WAV](male_cfg2_temp0.5.wav) |

The baseline WAVs are byte-identical to the `none` clips in the
[slurring diagnostic](../malayalam-slurring-diagnostic/README.md). All four
settings had the same sample count per voice: 7.520 s female and 7.275 s male.
That shows the settings did not change predicted timing; it does not tell us
whether articulation changed. The prepared text for both utterances was
unchanged by the Malayalam normalizer on a second pass. Listen for consonant
clarity and syllable separation; prefer an inference adjustment only if it
improves both voices without adding noise.

Exact inputs are in [manifest.json](manifest.json). Reproduce with
[`scripts/compare_malayalam_inference_settings.py`](../../../scripts/compare_malayalam_inference_settings.py).
