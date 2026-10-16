# Malayalam slurring diagnosis: matched sentences

The [previous unbatched samples](../malayalam-unbatched-listening-2026-10-16/README.md)
were generated through the normal FP32 `Synthesizer.__call__` path. Thus their
slurred delivery cannot be attributed to `Serving.fast()`, FP16, or batching.

This set uses the **same acoustic 150k and Vocos 110k checkpoints**, the same
text and voice for each row, and `cfg_scale=2`, `temperature=0.3`, and the
original sample's seed. Synthesis is again unbatched FP32. The `sampled` files
are **byte-identical** to female sample 01 and male sample 05 in the previous
folder.

Listen from left to right:

| Voice and source | Original recording | Recorded mel through Vocos | No prosody | Prosody pitch, regressor duration | Sampled prosody pitch and duration |
|---|---|---|---|---|---|
| Female `MAL_F_SAD_00441` | [8.11 s](female_recording.wav) | [8.10 s](female_recorded_mel_vocos.wav) | [7.52 s](female_none.wav) | [7.52 s](female_regressor.wav) | [8.25 s](female_sampled.wav) |
| Male `MAL_M_WIKI_00133` | [7.34 s](male_recording.wav) | [7.33 s](male_recorded_mel_vocos.wav) | [7.28 s](male_none.wav) | [7.28 s](male_regressor.wav) | [7.68 s](male_sampled.wav) |

Interpretation:

- If the recorded-mel Vocos copy already sounds slurred relative to the original,
  investigate the vocoder or mel representation.
- If the recorded-mel copy is clear but **no prosody** is slurred, investigate
  the acoustic model, text frontend, or acoustic checkpoint.
- If no prosody is clear but **regressor duration** is slurred, investigate the
  sampled prosody pitch.
- If regressor duration is clear but **sampled duration** slurs, use regressor
  durations while investigating the duration sampler.

Sampled durations made these two outputs 0.73 s (female) and 0.41 s (male)
longer than regressor durations. This is a pacing difference, not by itself a
proof of the slurring cause. The ten WAVs and exact source IDs are recorded in
[manifest.json](manifest.json). Reproduce with
[`scripts/diagnose_malayalam_slurring.py`](../../../scripts/diagnose_malayalam_slurring.py).
