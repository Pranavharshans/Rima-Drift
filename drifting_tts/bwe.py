"""Mel bandwidth extension (MelBWE): fill the band above a recording's cutoff in a BigVGAN log-mel.

A voice learned from 16 kHz recordings (resampled to 24 kHz by ``prepare``) has nothing above ~8 kHz in its mels, so
the vocoder renders it band-limited, dull next to the 24 kHz voices. :class:`MelBWE` is a small dilated convolution
net over time that predicts the whole log-mel from a band-limited one; it is trained on the full-band corpus, each
clip low-passed the way a 16 kHz source is (``drifting-tts train-bwe``). At inference the bins below the cutoff are
kept as generated and only the bins above it come from the net (:meth:`MelBWE.extend`), so the voice itself (the band
the model learned) is untouched. :func:`extend_vocoder` puts it in front of a :class:`~drifting_tts.vocoder.Vocoder`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import Tensor, nn

from .audio import N_MELS, SAMPLE_RATE

LOG_FLOOR = -11.5129  # log(1e-5): the BigVGAN front end's floor


def mel_centres(n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """Centre frequency (Hz) of each Slaney mel bin of the BigVGAN front end."""
    import torchaudio

    fb = torchaudio.functional.melscale_fbanks(1024 // 2 + 1, 0.0, sample_rate / 2, n_mels, sample_rate,
                                               norm="slaney", mel_scale="slaney")  # [freqs, mels]
    freqs = torch.linspace(0, sample_rate / 2, fb.shape[0])
    return (fb * freqs[:, None]).sum(0) / fb.sum(0).clamp_min(1e-9)


class Block(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.mlp = nn.Sequential(nn.Conv1d(dim, 2 * dim, 1), nn.GELU(), nn.Conv1d(2 * dim, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.conv(self.norm(x)))


class MelBWE(nn.Module):
    """Band-limited log-mel ``[B, n_mels, T]`` -> full-band log-mel (same scale). Receptive field: ±(sum of the
    dilations) frames, within a streaming vocoder window's context."""

    def __init__(self, n_mels: int = N_MELS, dim: int = 256, dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 1),
                 mean: float = -5.0, std: float = 2.2):
        super().__init__()
        self.config = {"n_mels": n_mels, "dim": dim, "dilations": list(dilations), "mean": mean, "std": std}
        self.mean, self.std = mean, std
        self.inp = nn.Conv1d(n_mels, dim, 3, padding=1)
        self.blocks = nn.ModuleList(Block(dim, d) for d in dilations)
        self.out = nn.Conv1d(dim, n_mels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("centres", mel_centres(n_mels), persistent=False)

    def forward(self, mel: Tensor) -> Tensor:
        x = (mel - self.mean) / self.std
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return mel + self.out(h) * self.std

    def weights(self, cutoff_hz: float, width: float = 400.0) -> Tensor:
        """Per-bin share of the net's output: 0 below ``cutoff_hz - width``, 1 above ``cutoff_hz``, linear between."""
        return ((self.centres - (cutoff_hz - width)) / width).clamp(0, 1)

    @torch.no_grad()
    def extend(self, mel: Tensor, cutoff_hz: float) -> Tensor:
        """``mel`` with the bins above ``cutoff_hz`` filled by the net, the ones below kept (crossfaded between)."""
        w = self.weights(cutoff_hz).to(mel)[:, None]
        return mel * (1 - w) + self(mel) * w

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> MelBWE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        m = cls(**{**cfg, "dilations": tuple(cfg["dilations"])})
        m.load_state_dict(ck["model"])
        m.cutoff_hz = float(ck.get("cutoff_hz", 7600.0))
        return m.to(device).eval()


def extend_vocoder(vocoder, bwe: MelBWE | str | Path, cutoff_hz: float | None = None):
    """A copy of ``vocoder`` (a :class:`~drifting_tts.vocoder.Vocoder`) that first fills the band above ``cutoff_hz``
    (default: the checkpoint's) with ``bwe``; e.g. ``synth.variant(vocoder=extend_vocoder(synth.vocoder, path))``."""
    if not isinstance(bwe, MelBWE):
        bwe = MelBWE.load(bwe, vocoder.device)
    other = copy.copy(vocoder)
    other.bwe, other.bwe_cutoff = bwe, float(cutoff_hz if cutoff_hz is not None else bwe.cutoff_hz)
    other.name = f"{vocoder.name or vocoder.kind}+bwe"
    return other
