"""Evaluate a separator on full-length Libri2Mix utterances (own implementation).

Writes per-utterance metrics + diagnostics to <out>/per_utt.csv and a summary to
<out>/summary.json. With --save_audio, also writes mixture / references / estimates.

  python evaluate.py --model srcorrnet --ckpt pretrained --out results/zeroshot
  python evaluate.py --model srcorrnet --ckpt runs/ft_full/best.pt --out results/ft_full --sdr --perceptual
  python evaluate.py --model srcorrnet --ckpt runs/ft_full/best.pt --out results/ft_full \
      --ids_file results/example_ids.txt --save_audio
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from common import SR, find_libri2mix, load_model, load_triplet, pit_si_sdr, separate, si_sdr


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/kaggle/input")
    ap.add_argument("--mix_type", default="mix_clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", choices=["srcorrnet", "baseline"], default="srcorrnet")
    ap.add_argument("--ckpt", default="pretrained")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_utts", type=int, default=None, help="evaluate a fixed random subset of N ids (seed 0)")
    ap.add_argument("--ids_file", default=None, help="only evaluate these ids (one per line)")
    ap.add_argument("--save_audio", action="store_true")
    ap.add_argument("--sdr", action="store_true", help="also BSS-eval SDR via mir_eval (slower)")
    ap.add_argument("--perceptual", action="store_true", help="also STOI and narrow-band PESQ")
    return ap.parse_args()


# Diagnostics
def active_mask(x: np.ndarray, frame: int = 256, thr_db: float = -30.0) -> np.ndarray:
    """Frame-level activity: energy within thr_db of the utterance's loudest frame."""
    n = len(x) // frame
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12
    return 10 * np.log10(e / e.max()) > thr_db


def median_f0(x: np.ndarray) -> float:
    import librosa
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=SR, frame_length=512, hop_length=256)
    act = active_mask(x, 256)
    m = min(len(f0), len(act))
    v = f0[:m][act[:m]]
    return float(np.median(v)) if len(v) else float("nan")


def swap_fraction(est: torch.Tensor, ref: torch.Tensor, win_s: float = 1.0, margin_db: float = 1.0) -> float:
    """Fraction of 1 s windows (where both speakers are active) whose locally best output
    assignment disagrees with the utterance-level PIT assignment by more than margin_db.
    Utterance-level PIT hides these mid-utterance swaps; this exposes them."""
    W = int(win_s * SR)
    n = ref.shape[-1] // W
    if n < 2:
        return float("nan")
    e = est[:, : n * W].reshape(2, n, W).transpose(0, 1)   # [n, 2, W]
    r = ref[:, : n * W].reshape(2, n, W).transpose(0, 1)
    en = r.pow(2).mean(-1)                                  # [n, 2]
    both = (10 * torch.log10(en / en.max(0).values + 1e-12) > -20).all(-1)
    keep = si_sdr(e, r).mean(-1)
    swap = si_sdr(e.flip(1), r).mean(-1)
    disagree = (swap - keep) > margin_db
    if both.sum() == 0:
        return float("nan")
    return (disagree & both).sum().item() / both.sum().item()


def bss_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    from mir_eval.separation import bss_eval_sources
    sdr, _, _, _ = bss_eval_sources(ref, est, compute_permutation=False)
    return float(sdr.mean())


def perceptual(est: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    from pystoi import stoi
    from pesq import pesq
    st = float(np.mean([stoi(ref[i], est[i], SR, extended=False) for i in range(2)]))
    try:
        pq = float(np.mean([pesq(SR, ref[i], est[i], "nb") for i in range(2)]))
    except Exception:  # PESQ raises on e.g. no detected speech
        pq = float("nan")
    return st, pq


def to_listening_level(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Rescale each estimate by its SI-SDR projection gain so it is level-matched to its
    reference (scale-invariant models output arbitrary gain)."""
    a = (est * ref).sum(-1, keepdim=True) / (est.pow(2).sum(-1, keepdim=True) + 1e-8)
    return est * a


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = find_libri2mix(a.data_root, a.mix_type)[a.split]
    ids = info["ids"]
    if a.ids_file:
        wanted = [l.strip() for l in open(a.ids_file) if l.strip()]
        ids = [i for i in wanted if i in set(ids)]
    elif a.max_utts:
        ids = sorted(random.Random(0).sample(ids, min(a.max_utts, len(ids))))  # same subset for every system
    print(f"{a.split}: evaluating {len(ids)} utterances from {info['dir']} ({info['sr']} Hz -> {SR} Hz)")

    model = load_model(a.model, a.ckpt).to(device).eval()
    rows = []
    for utt in tqdm(ids, mininterval=30):
        mix, ref = load_triplet(info, utt)
        t0 = time.time()
        est = separate(model, mix, device)
        rtf = (time.time() - t0) / (mix.shape[-1] / SR)
        best, swapped = pit_si_sdr(est[None], ref[None])
        est = est.flip(0) if swapped.item() else est          # now est[i] matches ref[i]
        sin = si_sdr(mix[None].expand_as(ref), ref)
        refn, estn, mixn = ref.numpy(), est.numpy(), mix.numpy()
        row = {
            "id": utt,
            "dur_s": mix.shape[-1] / SR,
            "si_sdr_in": sin.mean().item(),
            "si_sdr": best.mean().item(),
            "si_sdri": (best[0] - sin).mean().item(),
            "si_sdr_s1": best[0, 0].item(),
            "si_sdr_s2": best[0, 1].item(),
            "src_level_diff_db": 10 * np.log10((refn[0] ** 2).sum() / ((refn[1] ** 2).sum() + 1e-12) + 1e-12),
            "f0_s1": median_f0(refn[0]),
            "f0_s2": median_f0(refn[1]),
            "overlap": float((active_mask(refn[0]) & active_mask(refn[1])).mean()),
            "swap_frac": swap_fraction(est, ref),
            "rtf": rtf,
        }
        row["df0"] = abs(row["f0_s1"] - row["f0_s2"])
        if info.get("sex"):
            row["sex_pair"] = info["sex"].get(utt, "")  # "FF", "FM" or "MM" from LibriMix metadata
        if a.sdr:
            row["sdr"] = bss_sdr(estn, refn)
            row["sdri"] = row["sdr"] - bss_sdr(np.stack([mixn, mixn]), refn)
        if a.perceptual:
            row["stoi"], row["pesq"] = perceptual(estn, refn)
            row["stoi_in"], row["pesq_in"] = perceptual(np.stack([mixn, mixn]), refn)
        rows.append(row)

        if a.save_audio:
            d = out / "audio" / Path(utt).stem
            d.mkdir(parents=True, exist_ok=True)
            lis = to_listening_level(est, ref)
            peak = max(mix.abs().max(), ref.abs().max(), lis.abs().max()).item() / 0.95
            peak = max(peak, 1.0)
            sf.write(d / "mix.wav", mixn / peak, SR)
            for i in range(2):
                sf.write(d / f"s{i+1}_ref.wav", refn[i] / peak, SR)
                sf.write(d / f"s{i+1}_est.wav", lis[i].numpy() / peak, SR)

    df = pd.DataFrame(rows)
    name = "per_utt_examples.csv" if a.ids_file else "per_utt.csv"
    df.to_csv(out / name, index=False)
    if a.ids_file:
        print(f"saved {len(df)} examples to {out/'audio'}")
        return
    num = df.select_dtypes("number")
    summary = {"model": a.model, "ckpt": a.ckpt, "split": a.split, "n": len(df), "sr": SR,
               "mean": num.mean().round(3).to_dict(), "std": num.std().round(3).to_dict(),
               "median": num.median().round(3).to_dict(),
               "frac_si_sdri_below_5db": float((df.si_sdri < 5).mean()),
               "frac_swap_gt_0": float((df.swap_frac > 0).mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"SI-SDRi {df.si_sdri.mean():.2f} ± {df.si_sdri.std():.2f} dB (median {df.si_sdri.median():.2f}) "
          f"over {len(df)} utts" + (f" | SDRi {df.sdri.mean():.2f}" if a.sdr else "")
          + (f" | STOI {df.stoi.mean():.3f} PESQ {df.pesq.mean():.2f}" if a.perceptual else ""))


if __name__ == "__main__":
    main()
