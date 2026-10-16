# Unbatched Malayalam listening samples

Five fresh sentences from the `ai4bharat/Rasa` Malayalam dev split, generated
once per voice through `Synthesizer.__call__`. Each call processed **one request**
with FP32 inference, no batching, no `Serving.fast()`, no CUDA graphs, no
`torch.compile`, and no TF32. The checkpoints were the 150k acoustic model,
110k Vocos continuation, and 20k prosody model. Configuration: `cfg_scale=2`,
`temperature=0.3`, and the seed listed in [manifest.json](manifest.json).
All WAVs are mono 24 kHz PCM and at least four seconds long.

| # | Malayalam text | Female | Male |
|---|---|---|---|
| 1 | രേഖകള്‍ വേദനാജനകമായ ഒരു സത്യം വെളിപ്പെടുത്തിയപ്പോൾ മുറിയാകെ നിശബ്ദമായി, പ്രതീക്ഷകളുടെ തിളക്കം കെടുത്തി. | [8.25 s](01_female.wav) | [6.68 s](01_male.wav) |
| 2 | തീർച്ചയായും. ബ്രിട്ടീഷ് ഇന്ത്യ രാജ്യത്തുടനീളം ധാരാളം പള്ളികളുടെ നിർമ്മാണത്തിന് സാക്ഷ്യം വഹിച്ചു. | [6.42 s](02_female.wav) | [5.64 s](02_male.wav) |
| 3 | ദക്ഷിണ ഒഡീഷയിൽ, ഒരു മാസം നീണ്ടുനിൽക്കുന്ന ദണ്ഡ നട നൃത്തോത്സവത്തിന്റെ അവസാനമായാണ് മേരു യാത്രാ ഉത്സവം ആഘോഷിക്കുന്നത്. | [8.97 s](03_female.wav) | [7.66 s](03_male.wav) |
| 4 | ശുചിത്വം ഉറപ്പുവരുത്തുന്നതിന്, എല്ലാ വിദ്യാർത്ഥിനികളും കോയിൻ എടുക്കൽ ഞങ്ങൾ നിർബന്ധമാക്കിയിട്ടുണ്ട്. | [5.72 s](04_female.wav) | [5.24 s](04_male.wav) |
| 5 | രാഗത്തിൻ്റെ ഭംഗി വിശദീകരിക്കാൻ രചനയുടെ ഈണത്തിലും അതുപോലെ സ്വരങ്ങൾ ഉപയോഗിക്കുന്നതിലും തത്ക്ഷണ രചനാപാടവം സംഭവിക്കുന്നു. | [8.88 s](05_female.wav) | [7.68 s](05_male.wav) |

The reproducible generator is
[`scripts/generate_malayalam_unbatched_samples.py`](../../../scripts/generate_malayalam_unbatched_samples.py).
These are listening samples, not a formal quality evaluation.
