"""The Vocos ONNX graph of ``scripts/export_onnx.py`` (#8): the ISTFT head in plain operators against the
synthesizer's torch.fft path, and the exported graph under ONNX Runtime. Tiny random Vocos, CPU."""

import numpy as np
import pytest
import torch

pytest.importorskip("vocos")

from drifting_tts.vocoder import Vocoder  # noqa: E402
from scripts.export_onnx import VocosVocoder, export  # noqa: E402
from tests.test_vocoder_registry import _tiny_vocos  # noqa: E402


@pytest.mark.parametrize("padding", ["same", "center"])
def test_vocos_istft_in_plain_operators_matches_torch_fft(tmp_path, padding):
    model, _ = _tiny_vocos(tmp_path, padding=padding)
    for p in model.backbone.parameters():  # larger weights: a non-trivial spectrum
        p.data.normal_(0, 0.3)
    voc = Vocoder.wrap(model.eval(), "vocos", "cpu", "bigvgan", "tiny", 32)
    graph = VocosVocoder(model).eval()
    for t in (1, 7, 64):
        mel = torch.randn(1, 100, t) * 2 - 5
        with torch.no_grad():
            ref = voc(mel)
            out = graph(mel)
        assert out.shape == ref.shape == (1, t * 256)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


def test_vocos_onnx_export_runs_at_any_length(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, _ = _tiny_vocos(tmp_path, padding="same")
    graph = VocosVocoder(model).eval()
    path = tmp_path / "vocoder.onnx"
    export(graph, (torch.randn(1, 100, 61),), path, ["mel"], ["audio"], {"mel": {2: "T"}, "audio": {1: "S"}})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ops = {n.op_type for n in __import__("onnx").load(str(path)).graph.node}
    assert not ops & {"DFT", "STFT"}  # plain operators only (onnxruntime-web has no DFT on WebGPU)
    for t in (13, 97):
        mel = torch.randn(1, 100, t) * 2 - 5
        with torch.no_grad():
            ref = graph(mel).numpy()
        out = sess.run(None, {"mel": mel.numpy()})[0]
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-5)
