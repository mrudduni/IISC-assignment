"""Shared code for Part 1: data, losses, metrics, and model wrappers.

Own implementation unless noted. The SR-CorrNet network itself (``Model``) and its
STFT/iSTFT layers come from the official repo (github.com/dmlguq456/SR_CorrNet_SS, MIT).
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 8000  # the pretrained SR-CorrNet checkpoint operates at 8 kHz
HF_REPO = "shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk"
EPS = 1e-8


# Libri2Mix discovery
def _split_name(path: Path) -> str | None:
    for part in reversed(path.parts):
        p = part.lower()
        if p.startswith("train"):
            return "train"
        if p.startswith("dev") or p in ("valid", "val", "cv"):
            return "dev"
        if p.startswith("test") or p == "tt":
            return "test"
    return None


LIBRIMIX_SPLITS = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}


def _roots(root: str) -> list[str]:
    return [r for r in str(root).split(",") if r]


def _find_wav_splits(roots: list[str], mix_type: str) -> dict[str, dict]:
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root, followlinks=True):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.isdigit()]
            names = set(dirnames)
            if {"s1", "s2"} <= names and mix_type in names:
                d = Path(dirpath)
                split = _split_name(d)
                if split is None:
                    continue
                s = str(d).lower()
                score = (2 if "8k" in s else 0) + (1 if "min" in s else 0) + (1 if "100" in s else 0)
                candidates.setdefault(split, []).append((score, d))
    found = {}
    for split, cands in candidates.items():
        cands.sort(key=lambda x: -x[0])
        d = cands[0][1]
        mix_files = {p.name for p in (d / mix_type).glob("*.wav")}
        s1_files = {p.name for p in (d / "s1").glob("*.wav")}
        s2_files = {p.name for p in (d / "s2").glob("*.wav")}
        ids = sorted(mix_files & s1_files & s2_files)
        if not ids:
            continue
        sr = sf.info(str(d / mix_type / ids[0])).samplerate
        found[split] = {"kind": "wav", "dir": str(d), "sr": sr, "ids": ids, "mix_type": mix_type}
    return found


def _find_meta_splits(roots: list[str]) -> dict[str, dict]:
    """Libri2Mix built on the fly from LibriSpeech FLACs + the official LibriMix metadata CSVs.

    Uses ``metadata/Libri2Mix/libri2mix_<split>.csv`` (source paths and gains) exactly as the
    official generator does: sources * gain, truncated to the shorter one ("min" mode), summed.
    Noise columns are ignored (= ``mix_clean``). Speaker sex comes from ``*_info.csv``.
    """
    import pandas as pd
    meta, ls_dirs = None, {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            if meta is None and "libri2mix_test-clean.csv" in filenames:
                meta = Path(dirpath)
            for name in dirnames:
                if name in LIBRIMIX_SPLITS.values() and name not in ls_dirs:
                    cand = Path(dirpath) / name
                    subs = next(os.walk(cand), (None, [], None))[1]
                    if subs and all(x.isdigit() for x in subs[:5]):  # LibriSpeech speaker folders
                        ls_dirs[name] = cand
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.isdigit()]
    found = {}
    if meta is None:
        return found
    for split, ls_name in LIBRIMIX_SPLITS.items():
        csv = meta / f"libri2mix_{ls_name}.csv"
        if not csv.exists() or ls_name not in ls_dirs:
            continue
        base = ls_dirs[ls_name].parent  # metadata paths start with "<ls_name>/spk/chapter/..."
        df = pd.read_csv(csv)
        rows = {r.mixture_ID: (str(base / r.source_1_path), float(r.source_1_gain),
                               str(base / r.source_2_path), float(r.source_2_gain)) for r in df.itertuples()}
        sex = {}
        info_csv = meta / f"libri2mix_{ls_name}_info.csv"
        if info_csv.exists():
            di = pd.read_csv(info_csv)
            sex = {r.mixture_ID: "".join(sorted(str(r.speaker_1_sex) + str(r.speaker_2_sex))) for r in di.itertuples()}
        first = next(iter(rows.values()))[0]
        if not os.path.exists(first):
            raise FileNotFoundError(f"metadata points to {first}, which does not exist")
        found[split] = {"kind": "meta", "dir": str(ls_dirs[ls_name]), "sr": sf.info(first).samplerate,
                        "ids": sorted(rows), "rows": rows, "sex": sex, "mix_type": "mix_clean (on the fly)"}
    return found


def find_libri2mix(root: str, mix_type: str = "mix_clean") -> dict[str, dict]:
    """Return {split: info} for train/dev/test. ``root`` may be several comma-separated dirs.

    1. Generated Libri2Mix wavs: split folders (train-100/dev/test) containing ``<mix_type>/``, ``s1/``,
       ``s2/``; wav8k or wav16k, prefers wav8k and ``min``.
    2. Otherwise raw LibriSpeech (train-clean-100, dev-clean, test-clean) + LibriMix ``metadata/``:
       mixtures are created on the fly, identical to Libri2Mix ``min`` / ``mix_clean``.
    """
    roots = _roots(root)
    return _find_wav_splits(roots, mix_type) or _find_meta_splits(roots)


# Audio I/O
def resample(x: torch.Tensor, sr_in: int, sr_out: int = SR) -> torch.Tensor:
    """Polyphase resampling (scipy), e.g. Libri2Mix wav16k -> 8 kHz. Works on [..., L]."""
    if sr_in == sr_out:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    y = resample_poly(x.numpy(), sr_out // g, sr_in // g, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def load_wav(path: str, start: int = 0, frames: int = -1) -> tuple[torch.Tensor, int]:
    x, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    return torch.from_numpy(x[:, 0].copy()), sr


def load_triplet(split_info: dict, utt_id: str, start_s: float | None = None,
                 dur_s: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mix [L], sources [2, L]) at 8 kHz. Optional crop given in seconds."""
    sr = split_info["sr"]
    start = 0 if start_s is None else int(start_s * sr)
    frames = -1 if dur_s is None else int(dur_s * sr)
    if split_info["kind"] == "meta":
        p1, g1, p2, g2 = split_info["rows"][utt_id]
        s1, _ = load_wav(p1, start, frames)
        s2, _ = load_wav(p2, start, frames)
        L = min(len(s1), len(s2))
        srcs = resample(torch.stack([s1[:L] * g1, s2[:L] * g2]), sr)
        return srcs.sum(0), srcs
    d, mt = Path(split_info["dir"]), split_info["mix_type"]
    mix, _ = load_wav(str(d / mt / utt_id), start, frames)
    s1, _ = load_wav(str(d / "s1" / utt_id), start, frames)
    s2, _ = load_wav(str(d / "s2" / utt_id), start, frames)
    L = min(len(mix), len(s1), len(s2))
    mix, srcs = mix[:L], torch.stack([s1[:L], s2[:L]])
    mix, srcs = resample(mix, sr), resample(srcs, sr)
    return mix, srcs


def utt_frames(split_info: dict, utt_id: str) -> int:
    """Length in samples at the source rate (shorter source for on-the-fly mixtures)."""
    if split_info["kind"] == "meta":
        p1, _, p2, _ = split_info["rows"][utt_id]
        return min(sf.info(p1).frames, sf.info(p2).frames)
    return sf.info(str(Path(split_info["dir"]) / split_info["mix_type"] / utt_id)).frames


class Libri2MixDataset(torch.utils.data.Dataset):
    """Random fixed-length crops for training, deterministic crops for validation."""

    def __init__(self, split_info: dict, segment_s: float = 4.0, train: bool = True,
                 max_items: int | None = None, seed: int = 0):
        self.info = split_info
        self.ids = list(split_info["ids"])
        if max_items is not None and max_items < len(self.ids):
            rng = random.Random(seed)
            self.ids = sorted(rng.sample(self.ids, max_items))
        self.segment_s = segment_s
        self.train = train
        self.seg = int(segment_s * SR)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        utt = self.ids[i]
        sr = self.info["sr"]
        dur = utt_frames(self.info, utt) / sr
        if dur > self.segment_s:
            start = random.uniform(0, dur - self.segment_s) if self.train else 0.0
            mix, srcs = load_triplet(self.info, utt, start, self.segment_s)
        else:
            mix, srcs = load_triplet(self.info, utt)
        mix, srcs = mix[: self.seg], srcs[:, : self.seg]
        pad = self.seg - mix.shape[-1]
        if pad > 0:  # zero-pad short utterances; zeros are harmless for SI-SDR
            mix = F.pad(mix, (0, pad))
            srcs = F.pad(srcs, (0, pad))
        if self.train:  # random gain: model and loss are scale-invariant, but this de-biases levels
            g = 10 ** (random.uniform(-5, 5) / 20)
            mix, srcs = mix * g, srcs * g
        return mix, srcs


# Losses / metrics (own implementation)
def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB over the last axis. Shapes broadcast."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + EPS) + EPS)


def pit_si_sdr(est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Utterance-level PIT for 2 sources. est, ref: [B, 2, L].

    Returns (per-source SI-SDR of the best permutation [B, 2], swapped flag [B]).
    """
    keep = si_sdr(est, ref)                       # [B, 2]
    swap = si_sdr(est.flip(1), ref)               # [B, 2]
    use_swap = swap.mean(-1) > keep.mean(-1)
    best = torch.where(use_swap[:, None], swap, keep)
    return best, use_swap


def apply_perm(est: torch.Tensor, swapped: torch.Tensor) -> torch.Tensor:
    return torch.where(swapped[:, None, None], est.flip(1), est)


def mag_snr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Negative SNR between STFT magnitudes (phase-insensitive). Used for SR-CorrNet's
    intermediate decoder outputs, mirroring the auxiliary loss in the official repo."""
    win = torch.hann_window(512, device=est.device)
    def mag(x):
        X = torch.stft(x.reshape(-1, x.shape[-1]), 512, 256, window=win, return_complex=True)
        return X.abs()
    e, r = mag(est), mag(ref)
    snr = 10 * torch.log10(r.pow(2).sum((-1, -2)) / ((e - r).pow(2).sum((-1, -2)) + EPS) + EPS)
    return -snr.clamp(max=30).mean()


# Models
def _load_hf_files(repo_id: str = HF_REPO) -> tuple[dict, str]:
    import yaml
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["config"]
    return cfg, ckpt_path


class SRCorrNetSeparator(nn.Module):
    """Waveform-in / waveform-out wrapper around the official SR-CorrNet ``Model``.

    Mirrors the official engine: mixture is std-normalised, transformed with the repo's
    STFT (128/64 at 8 kHz), fed as stacked real/imag, and the estimated complex STFTs
    are inverted with the repo's iSTFT.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        from sr_corrnet.models.SR_CorrNet_SS.model import Model
        from sr_corrnet.utils.util_stft import STFT, iSTFT
        self.cfg = cfg
        self.net = Model(**cfg["model"])
        # build on CPU so the kernels are registered buffers/params and follow .to(device)
        self.stft = STFT(**cfg["stft"], device="cpu", normalize=True)
        self.istft = iSTFT(**cfg["stft"], device="cpu", normalize=True)

    @classmethod
    def pretrained(cls, repo_id: str = HF_REPO) -> "SRCorrNetSeparator":
        from sr_corrnet.utils.util_engine import _fix_compiled_state_dict, _migrate_state_dict
        cfg, ckpt_path = _load_hf_files(repo_id)
        m = cls(cfg)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
        sd = _migrate_state_dict(_fix_compiled_state_dict(sd))
        # the released file also carries thop profiler counters (total_ops/total_params); drop them
        sd = {k: v for k, v in sd.items() if not k.endswith(("total_ops", "total_params"))}
        missing, unexpected = m.net.load_state_dict(sd, strict=False)
        n = len(m.net.state_dict())
        print(f"[SR-CorrNet] loaded {repo_id}: {n - len(missing)}/{n} tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            raise RuntimeError(f"Pretrained checkpoint is missing weights: {missing[:5]} ...")
        return m

    def enable_grad_checkpointing(self):
        """Recompute each heavy block in the backward pass instead of storing its activations.
        Training memory drops from ~4.2 GiB per audio-second to roughly one block's worth, for
        ~30% more compute. Outputs and gradients are unchanged."""
        from torch.utils.checkpoint import checkpoint
        mods = [self.net.encoder, self.net.filter_estim, *self.net.enc_block, *self.net.dec_block,
                *self.net.dec_cs, *self.net.filter_estim_aux]
        for mod in mods:
            fwd = mod.forward
            def wrapped(*args, _fwd=fwd, **kw):
                if torch.is_grad_enabled():
                    return checkpoint(_fwd, *args, use_reentrant=False, **kw)
                return _fwd(*args, **kw)
            mod.forward = wrapped
        return self

    def encoder_params(self):
        return list(self.net.encoder.parameters()) + list(self.net.enc_block.parameters())

    def _to_wave(self, outs: list, L: int) -> torch.Tensor:
        waves = []
        for e in outs:  # e: [B, M_o, F, T, 2]
            c = torch.complex(e[..., 0], e[..., 1]).squeeze(1)  # [B, F, T]
            w = self.istft(c, cplx=True)
            w = w.reshape(w.shape[0], -1)
            w = w[:, :L] if w.shape[-1] >= L else F.pad(w, (0, L - w.shape[-1]))
            waves.append(w)
        return torch.stack(waves, 1)  # [B, N, L]

    def forward(self, mix: torch.Tensor, aux: bool = False):
        """mix: [B, L] -> est [B, 2, L] (and list of aux estimates if aux=True)."""
        L = mix.shape[-1]
        x = mix.unsqueeze(1)                                  # [B, 1, L]
        X = self.stft(x, cplx=True)                           # [B, 1, F, T]
        inp = torch.cat([X.real, X.imag], dim=1)              # [B, 2, F, T]
        out, out_aux, _ = self.net(inp, aux_loss=aux)
        est = self._to_wave(out, L)
        if aux:
            return est, [self._to_wave(o, L) for o in out_aux]
        return est


class BLSTMMaskNet(nn.Module):
    """From-scratch baseline (own implementation): STFT -> log-magnitude -> 3-layer BLSTM
    -> 2 sigmoid masks applied to the complex mixture STFT -> iSTFT. Trained end-to-end
    with the same waveform PIT SI-SDR loss as SR-CorrNet."""

    def __init__(self, n_fft: int = 256, hop: int = 64, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        F_ = n_fft // 2 + 1
        self.norm = nn.LayerNorm(F_)
        self.rnn = nn.LSTM(F_, hidden, layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(2 * hidden, 2 * F_)

    def forward(self, mix: torch.Tensor, aux: bool = False):
        L = mix.shape[-1]
        X = torch.stft(mix, self.n_fft, self.hop, window=self.win, return_complex=True)  # [B, F, T]
        feat = torch.log1p(X.abs()).transpose(1, 2)                                       # [B, T, F]
        h, _ = self.rnn(self.norm(feat))
        m = torch.sigmoid(self.proj(h)).view(h.shape[0], h.shape[1], 2, -1).permute(0, 2, 3, 1)  # [B,2,F,T]
        S = m * X.unsqueeze(1)
        est = torch.istft(S.reshape(-1, *S.shape[2:]), self.n_fft, self.hop, window=self.win, length=L)
        est = est.view(mix.shape[0], 2, L)
        return (est, []) if aux else est


def build_model(name: str, init: str = "pretrained") -> nn.Module:
    if name == "srcorrnet":
        if init == "pretrained":
            return SRCorrNetSeparator.pretrained()
        cfg, _ = _load_hf_files()
        return SRCorrNetSeparator(cfg)
    if name == "baseline":
        return BLSTMMaskNet()
    raise ValueError(name)


def load_model(name: str, ckpt: str) -> nn.Module:
    """ckpt = 'pretrained' (zero-shot SR-CorrNet) or a path saved by train.py."""
    if ckpt == "pretrained":
        return build_model(name, "pretrained")
    model = build_model(name, "scratch")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    model.load_state_dict(state)
    return model


# Inference on arbitrary-length audio
@torch.inference_mode()
def separate(model: nn.Module, mix: torch.Tensor, device: torch.device,
             max_full_s: float = 20.0, chunk_s: float = 8.0, overlap_s: float = 2.0) -> torch.Tensor:
    """mix: [L] at 8 kHz -> est [2, L]. Runs the whole file at once when it fits; otherwise
    falls back to overlapping chunks whose output order is aligned to the previous chunk by
    correlation on the overlap (chunk-wise processing reintroduces the permutation problem)."""
    model.eval()
    scale = mix.std() + EPS
    x = (mix / scale).to(device)
    L = x.shape[-1]
    if L <= max_full_s * SR:
        try:
            return model(x[None])[0].cpu() * scale
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    C, O = int(chunk_s * SR), int(overlap_s * SR)
    hop = C - O
    out = torch.zeros(2, L)
    weight = torch.zeros(L)
    fade = torch.ones(C)
    ramp = torch.linspace(0, 1, O)
    fade[:O], fade[-O:] = ramp, ramp.flip(0)
    prev_tail = None
    for start in range(0, max(L - O, 1), hop):
        seg = x[start:start + C]
        n = seg.shape[-1]
        est = model(F.pad(seg, (0, C - n))[None])[0, :, :n].cpu()
        if prev_tail is not None:
            head = est[:, :O]
            keep = (head * prev_tail).sum()
            swap = (head.flip(0) * prev_tail).sum()
            if swap > keep:
                est = est.flip(0)
        w = fade[:n].clone()
        if start == 0:
            w[:O] = 1
        if start + C >= L:
            w[-min(O, n):] = 1
        out[:, start:start + n] += est * w
        weight[start:start + n] += w
        prev_tail = est[:, -O:] if n == C else None
    return out / weight.clamp(min=EPS) * scale


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the Libri2Mix splits found under a root dir.")
    ap.add_argument("--data_root", default="/kaggle/input")
    ap.add_argument("--mix_type", default="mix_clean")
    a = ap.parse_args()
    found = find_libri2mix(a.data_root, a.mix_type)
    if not found:
        raise SystemExit(f"No Libri2Mix split with {a.mix_type}/, s1/, s2/ found under {a.data_root}")
    for k, v in found.items():
        print(f"{k:6s} {len(v['ids']):6d} mixtures  {v['sr']} Hz  {v['mix_type']}  {v['dir']}")
