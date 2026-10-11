"""ARDrift: an autoregressive acoustic model with a one-step drifting head (#42, the Pocket TTS recipe, our own model).

The same text side as :class:`~drifting_tts.models.tts.DriftingTTS` (character encoder, monotonic alignment against a
Gaussian prior, duration predictor), but the mel is generated **left to right**: a causal transformer runs over
patches of ``patch`` mel frames (4 frames = 43 ms), and at every step a small head turns noise into the next patch
in **one** pass, trained with the drifting field (:func:`~drifting_tts.drift.kyutai_drift_loss`, Kyutai's learned
temperature). The backbone sees the previous patches (teacher-forced, noise-augmented in training) and the
frame-aligned text condition of the current and the next ``lookahead`` patches; nothing tells it the pitch, so the
intonation is the model's own. Like Pocket TTS (an autoregressive backbone over continuous latents with a one-step
sampler head), but on this repository's mels (decoded by Vocos v2), with aligned text instead of a text prefix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..alignment import sequence_mask
from ..text import SYMBOLS
from .generator import RMSNorm, SwiGLU
from .text_encoder import TextEncoder, durations_to_alignment, expand, frames_to_alignment, rope


class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cache: dict | None = None) -> Tensor:
        """``x`` ``[B, L, D]``. Without ``cache``: causal attention over the sequence. With ``cache`` (a dict per layer,
        inference): ``x`` are the next ``L`` positions, appended to the cached keys and values."""
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        offset = 0 if cache is None else cache.get("n", 0)
        if cache is not None and offset:
            assert L == 1, "a cached step takes one position at a time"
        q, k = rope(self.q_norm(q), offset), rope(self.k_norm(k), offset)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"], cache["n"] = k, v, offset + L
            y = F.scaled_dot_product_attention(q, k, v, is_causal=L > 1)  # L > 1 only from an empty cache
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = CausalAttention(dim, heads)
        self.mlp = SwiGLU(dim, (int(2 / 3 * dim * mlp_ratio) + 31) // 32 * 32)

    def forward(self, x: Tensor, cache: dict | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class HeadBlock(nn.Module):
    """Residual MLP block modulated by the backbone state (adaLN-zero)."""

    def __init__(self, dim: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Linear(cond, 3 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale, gate = self.ada(c).chunk(3, -1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class DriftHead(nn.Module):
    """``(state [N, cond], noise [N, noise_dim]) -> patch residual [N, out]`` in one pass."""

    def __init__(self, cond: int, out: int, dim: int = 512, depth: int = 6, noise_dim: int | None = None):
        super().__init__()
        self.noise_dim = noise_dim or out
        self.inp = nn.Linear(self.noise_dim, dim)
        self.cond_in = nn.Linear(cond, dim)
        self.cond_act = nn.SiLU()
        self.blocks = nn.ModuleList([HeadBlock(dim, cond) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        c = self.cond_act(s)
        x = self.inp(z) + self.cond_in(s)
        for b in self.blocks:
            x = b(x, c)
        return self.out(self.norm(x))


def patchify(x: Tensor, k: int) -> Tensor:
    """``[B, C, T]`` (T a multiple of ``k``) -> ``[B, T / k, C * k]`` (frame-major inside a patch)."""
    B, C, T = x.shape
    return x.view(B, C, T // k, k).permute(0, 2, 3, 1).reshape(B, T // k, k * C)


def unpatchify(p: Tensor, k: int, c: int) -> Tensor:
    """Inverse of :func:`patchify`: ``[B, S, k * C]`` -> ``[B, C, S * k]``."""
    B, S, _ = p.shape
    return p.view(B, S, k, c).permute(0, 3, 1, 2).reshape(B, c, S * k)


class ARDriftTTS(nn.Module):
    """See the module docstring. ``cfg``: ``model`` section (``text``, ``ar``)."""

    def __init__(self, cfg, num_speakers: int, n_mels: int = 100):
        super().__init__()
        t, a = cfg.text, cfg.ar
        self.encoder = TextEncoder(len(SYMBOLS), n_mels=n_mels, d=t.d, heads=t.heads, layers=t.layers, ffn=t.ffn,
                                   dropout=t.dropout, num_speakers=num_speakers, spk_dim=t.spk_dim)
        self.n_mels, self.patch, self.lookahead = n_mels, int(a.patch), int(a.get("lookahead", 2))
        self.pitch_enabled = False  # the intonation is the autoregressive model's own
        k, cond_ch, dim = self.patch, n_mels + t.d, int(a.dim)
        self.x_in = nn.Linear(k * n_mels, dim)
        self.c_in = nn.Linear(k * cond_ch * (1 + self.lookahead), dim)
        self.spk = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.spk.weight, std=0.02)
        self.bos = nn.Parameter(torch.zeros(k * n_mels))
        self.blocks = nn.ModuleList([Block(dim, int(a.heads), float(a.get("mlp_ratio", 4.0)))
                                     for _ in range(int(a.depth))])
        self.norm = RMSNorm(dim)
        self.head = DriftHead(dim, k * n_mels, int(a.head_dim), int(a.head_depth), a.get("noise_dim"))
        self.duration_scale, self.duration_scales, self.temperature = 1.0, {}, None

    # ----------------------------------------------------------------------------------------------- conditioning
    def cond_patches(self, cond: Tensor) -> Tensor:
        """Aligned condition ``[B, C, T]`` (T a multiple of ``patch``) -> per-step inputs ``[B, S, C * k * (1 + L)]``:
        each step's own condition and that of the next ``lookahead`` steps (zeros past the end)."""
        p = patchify(cond, self.patch)
        if not self.lookahead:
            return p
        ahead = [F.pad(p[:, i:], (0, 0, 0, i)) for i in range(1, self.lookahead + 1)]
        return torch.cat([p, *ahead], -1)

    def backbone(self, x_prev: Tensor, cpatch: Tensor, spk: Tensor, caches: list[dict] | None = None) -> Tensor:
        """Previous patches ``[B, S, k * n_mels]`` and condition inputs ``[B, S, *]`` -> states ``[B, S, dim]``."""
        h = self.x_in(x_prev) + self.c_in(cpatch) + self.spk(spk)[:, None]
        for i, blk in enumerate(self.blocks):
            h = blk(h, None if caches is None else caches[i])
        return self.norm(h)

    def text(self, text: Tensor, text_len: Tensor, spk: Tensor):
        return self.encoder(text, text_len, spk)

    @staticmethod
    def frame_condition(h: Tensor, mu: Tensor, attn: Tensor) -> Tensor:
        return torch.cat([expand(mu, attn), expand(h, attn)], 1)

    def pad_frames(self, x: Tensor) -> Tensor:
        return F.pad(x, (0, (-x.shape[-1]) % self.patch))

    # -------------------------------------------------------------------------------------------------- training
    def teacher_forced(self, cond: Tensor, y: Tensor, spk: Tensor, input_noise: Tensor | None = None) -> Tensor:
        """States ``[B, S, dim]`` for every step, the previous target patches as input (plus ``input_noise`` ``[B]``
        times unit noise: robustness to the model's own, imperfect patches at inference)."""
        yp = patchify(y, self.patch)
        x_prev = torch.cat([self.bos.expand(yp.shape[0], 1, -1), yp[:, :-1]], 1)
        if input_noise is not None:
            x_prev = x_prev + input_noise[:, None, None] * torch.randn_like(x_prev)
        return self.backbone(x_prev, self.cond_patches(cond), spk)

    # ------------------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, cond: Tensor, spk: Tensor, temperature: float = 1.0, generator: torch.Generator | None = None,
                 prefix: Tensor | None = None) -> Tensor:
        """Aligned condition ``[B, C, T]`` -> normalised mel ``[B, n_mels, T]``, one step (patch) at a time with a KV
        cache. ``prefix``: known leading mel frames ``[B, n_mels, Tp]`` (``Tp`` a multiple of ``patch``), kept."""
        T = cond.shape[-1]
        cond = self.pad_frames(cond)
        cp = self.cond_patches(cond)
        B, S = cp.shape[:2]
        k, mu = self.patch, patchify(cond[:, : self.n_mels], self.patch)
        caches = [{} for _ in self.blocks]
        prev = self.bos.expand(B, 1, -1)
        out = []
        n_pre = 0 if prefix is None else prefix.shape[-1] // k
        pre = None if prefix is None else patchify(prefix, k)
        for t in range(S):
            s = self.backbone(prev, cp[:, t: t + 1], spk, caches)[:, 0]
            if t < n_pre:
                x = pre[:, t]
            else:
                z = torch.randn(B, self.head.noise_dim, device=cond.device, generator=generator) * temperature
                x = mu[:, t] + self.head(s, z)
            out.append(x)
            prev = x[:, None]
        mel = unpatchify(torch.stack(out, 1), k, self.n_mels)
        return mel[..., :T]

    @torch.no_grad()
    def aligned_condition(self, text: Tensor, text_len: Tensor, spk: Tensor, length_scale: float = 1.0,
                          pitch_shift: float = 0.0, durations: Tensor | None = None, pitch: Tensor | None = None,
                          context: int = 0) -> tuple[Tensor, Tensor]:
        if pitch is not None or pitch_shift:
            raise ValueError("ARDrift models generate their own intonation: no token pitch or pitch shift")
        h, mu, logw, x_mask = self.text(text, text_len, spk)
        if context:
            h, mu, logw, x_mask = (t[..., context:] for t in (h, mu, logw, x_mask))
            durations = None if durations is None else durations[:, context:]
        if durations is None:
            attn, y_len = durations_to_alignment(logw, x_mask, length_scale)
        else:
            attn, y_len = frames_to_alignment(durations, x_mask)
        return self.frame_condition(h, mu, attn), y_len

    @torch.no_grad()
    def synthesize(self, text: Tensor, text_len: Tensor, spk: Tensor, cfg_scale: float = 1.0,
                   temperature: float = 1.0, length_scale: float = 1.0, noise_labels: Tensor | None = None,
                   generator: torch.Generator | None = None, attn_window: int | None = None,
                   pitch_shift: float = 0.0, steps: int | None = None, durations: Tensor | None = None,
                   pitch: Tensor | None = None, context: int = 0) -> tuple[Tensor, Tensor]:
        """As :meth:`DriftingTTS.synthesize` (the arguments a DiT needs are accepted and ignored): normalised mels
        ``[B, n_mels, T]`` and their lengths. Durations from the duration predictor (``length_scale``) or given."""
        cond, y_len = self.aligned_condition(text, text_len, spk, length_scale, pitch_shift, durations, pitch, context)
        mel = self.generate(cond, spk, temperature, generator)
        return mel * sequence_mask(y_len, mel.shape[-1])[:, None], y_len


def ar_features(x: Tensor, k: int, n_mels: int, detail: bool = True) -> dict[str, Tensor]:
    """Drift kernel features of patches ``[R, S, k * n_mels]`` (R rows, S samples): ``patch`` (the whole patch, one
    location), ``frame`` (each frame a location), ``detail`` (each frame minus its frequency-smoothed envelope: the
    harmonic structure). Every map is ``[R, S, L, D]``."""
    from .mae import spectral_detail

    R, S, _ = x.shape
    frames = x.view(R, S, k, n_mels)
    out = {"patch": x[:, :, None], "frame": frames}
    if detail:
        d = spectral_detail(frames.reshape(R * S, k, n_mels).transpose(1, 2)).transpose(1, 2)
        out["detail"] = d.reshape(R, S, k, n_mels)
    return out


