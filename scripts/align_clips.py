"""Re-cut clips at their words: MMS forced alignment (torchaudio MMS_FA) of the normalised transcript, the clip
trimmed to [first word start - lead, last word end + trail], mels recomputed. Clips whose first or last word aligns
badly (score below --min-score: the word is cut off in the audio) are dropped. Writes data/<root>_fa (mels, index,
speakers, stats copied from the source; no audio / F0). Each kept line has ``src_i``, its line in the source
index, so the source's ``scores.jsonl`` still applies (``LIMIT=n``: the first n lines only).

    python scripts/align_clips.py data/voice1 [data/voice2 ...]   # -> data/voice1_fa, ...
"""
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torchaudio

from drifting_tts.audio import BigVGANLogMel

LEAD, TRAIL, MIN_SCORE = 0.08, 0.12, 0.30
TR = str.maketrans({"ç": "c", "ş": "s", "ğ": "g", "ı": "i", "ö": "o", "ü": "u", "â": "a", "î": "i", "û": "u"})
dev = "cuda" if torch.cuda.is_available() else "cpu"
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(dev).eval()
labels = bundle.get_labels(star=None)
dictionary = {c: i for i, c in enumerate(labels)}
logmel = BigVGANLogMel().to(dev)


def romanise(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text.lower()).translate(TR)
    t = "".join(c if c in dictionary or c == " " else " " for c in t)
    return [w for w in t.split() if w and w != "'"]


@torch.no_grad()
def align(w16: torch.Tensor, words: list[str]):
    emission, _ = model(w16[None].to(dev))
    tokens = [dictionary[c] for w in words for c in w]
    if emission.shape[1] < len(tokens) + 1:
        return None
    path, scores = torchaudio.functional.forced_align(emission, torch.tensor([tokens], device=dev), blank=0)
    spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp())
    ratio = w16.shape[0] / emission.shape[1]
    out, k = [], 0
    for w in words:
        s = spans[k: k + len(w)]
        k += len(w)
        out.append((s[0].start * ratio / 16000, s[-1].end * ratio / 16000,
                    sum(x.score * (x.end - x.start) for x in s) / sum(x.end - x.start for x in s)))
    return out


def run(root: str) -> None:
    src = Path(root)
    dst = src.with_name(f"{src.name}_fa")
    dst.mkdir(exist_ok=True)
    for f in ("speakers.json", "stats.json"):
        shutil.copy(src / f, dst / f)
    audio = np.memmap(src / "audio.bin", dtype=np.float16, mode="r")
    idx = [json.loads(line) for line in open(src / "index.jsonl")]
    for i, e in enumerate(idx):
        e["src_i"] = i  # line of the source root (its scores.jsonl)
    idx = idx[: int(os.environ.get("LIMIT", 0)) or None]
    kept = dropped = 0
    offset = 0
    lead_cut, trail_cut = [], []
    with open(dst / "mels.bin", "wb") as mels, open(dst / "index.jsonl", "w") as index:
        for e in idx:
            w = torch.from_numpy(np.asarray(audio[e["audio_offset"]: e["audio_offset"] + e["audio_samples"]],
                                            dtype=np.float32))
            words = romanise(e["norm_text"])
            spans = align(torchaudio.functional.resample(w, 24000, 16000), words) if words else None
            if e["split"] == "train" and (spans is None or spans[0][2] < MIN_SCORE or spans[-1][2] < MIN_SCORE):
                dropped += 1
                continue
            if spans is not None:
                a = max(0, int((spans[0][0] - LEAD) * 24000))
                b = min(len(w), int((spans[-1][1] + TRAIL) * 24000))
                if b - a < 12000:  # a degenerate alignment: drop a training clip, keep a held-out one whole
                    if e["split"] == "train":
                        dropped += 1
                        continue
                    a, b = 0, len(w)
                lead_cut.append(a / 24000)
                trail_cut.append((len(w) - b) / 24000)
                w = w[a:b]
            m = logmel(w.to(dev)[None])[0].T.cpu().numpy().astype(np.float16)  # [frames, 100]
            mels.write(m.tobytes())
            e = {k: v for k, v in e.items() if k not in ("audio_offset", "audio_samples")}
            e.update(offset=offset, frames=int(m.shape[0]), audio_seconds=round(len(w) / 24000, 3))
            offset += m.shape[0]
            index.write(json.dumps(e, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{src.name}: kept {kept}, dropped {dropped} (first/last word score < {MIN_SCORE}); cut at the start: median "
          f"{np.median(lead_cut):.2f} s, >0.1 s in {100 * np.mean(np.array(lead_cut) > 0.1):.0f}%; at the end: median "
          f"{np.median(trail_cut):.2f} s", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        run(r)
