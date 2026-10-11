"""Mel bandwidth extension: the net, the kept band, the vocoder hook and train-bwe on a tiny prepared dataset."""

import json

import torch

from drifting_tts.bwe import MelBWE, extend_vocoder, mel_centres


def test_mel_centres_and_kept_band():
    c = mel_centres()
    assert c.shape == (100,) and torch.all(c[1:] > c[:-1]) and 11000 < c[-1] < 12000
    torch.manual_seed(0)
    m = MelBWE(dim=16, dilations=(1, 2)).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    mel = torch.randn(2, 100, 50) - 5
    out = m.extend(mel, 7600.0)
    low, high = c < 7200, c >= 7600
    assert torch.equal(out[:, low], mel[:, low])  # the generated band is untouched
    torch.testing.assert_close(out[:, high], m(mel)[:, high])
    assert m(mel).shape == mel.shape


def test_extend_vocoder_and_train_bwe_cli(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.vocoder import load_vocoder
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "3", "--no-trim", "--backend", "bigvgan", "--save-audio"])
    n = sum(1 for _ in open(data / "index.jsonl"))
    (data / "scores.jsonl").write_text("".join(json.dumps({"i": i, "bw_hz": 12000.0}) + "\n" for i in range(n)))
    work = tmp_path / "bwe"
    main(["train-bwe", "--data", str(data), "--workdir", str(work), "--steps", "3", "--batch", "2", "--seconds", "0.5",
          "--dim", "16", "--val", "2", "--log-every", "1", "--workers", "0", "--device", "cpu"])
    bwe = MelBWE.load(work / "bwe.pt")
    assert bwe.cutoff_hz == 7600.0
    voc = load_vocoder("griffin-lim", "cpu", backend="bigvgan")
    voc.model.n_iter = 2
    wide = extend_vocoder(voc, work / "bwe.pt", 7000.0)
    assert wide.bwe_cutoff == 7000.0 and voc.bwe is None and wide.name.endswith("+bwe")
    mel = torch.randn(1, 100, 20) - 5
    assert wide(mel).shape == voc(mel).shape
