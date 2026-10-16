# Malayalam acoustic checkpoint listening comparison

The original recording and its reconstruction through Vocos sounded clear to
the user, while synthesized speech sounded slurred even with prosody disabled.
This comparison holds the **same text, source voice, seed, 110k Vocos checkpoint,
and unbatched FP32 inference** fixed while changing only the acoustic checkpoint.
Prosody is disabled throughout. The 150k WAVs are byte-identical to the
`none` clips in the [slurring diagnostic](../malayalam-slurring-diagnostic/README.md).

| Voice | 30k | 70k | 110k | 150k |
|---|---|---|---|---|
| Female, `MAL_F_SAD_00441` | [7.24 s](female_30000.wav) | [7.48 s](female_70000.wav) | [7.39 s](female_110000.wav) | [7.52 s](female_150000.wav) |
| Male, `MAL_M_WIKI_00133` | [6.91 s](male_30000.wav) | [7.17 s](male_70000.wav) | [7.00 s](male_110000.wav) | [7.28 s](male_150000.wav) |

The [original recording and Vocos reconstruction](../malayalam-slurring-diagnostic/README.md)
are the reference for each voice. Listen for word articulation, timing, and
background noise separately. If an earlier checkpoint is clearer, the acoustic
training trajectory needs review before choosing a final checkpoint. If every
checkpoint slurs, investigate the text frontend, alignments, and acoustic
model itself. The durations alone do not establish speech quality.

Exact inputs and file names are in [manifest.json](manifest.json). Reproduce
with [`scripts/compare_malayalam_acoustic_checkpoints.py`](../../../scripts/compare_malayalam_acoustic_checkpoints.py).
