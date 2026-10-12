"""Extra statistics and figures used in the report (own implementation, runs on CPU).

  python extra_analysis.py --results ../part1_results/results --data_root /path/to/LibriMix \
      [--rumble_audio ../path/to/zeroshot/audio/1580-141084-0024_2094-142345-0025]

Writes <results>/report/extra/: stats.md, low_freq_share.csv, fig_low_freq.png, fig_rumble_example.png.
Needs the LibriSpeech test-clean audio to measure how much reference energy lies below 100 Hz.
"""
import argparse
from pathlib import Path

from common import find_libri2mix, load_triplet  # imports torch; keep before matplotlib (Windows DLL order)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr, wilcoxon

from analyze import SERIES, INK2  # same style and colours as the main report


def low_freq_share(src: np.ndarray, sr: int = 8000, fc: float = 100.0) -> float:
    """Largest share of a reference's energy below fc Hz (over the two sources)."""
    x = src - src.mean(-1, keepdims=True)
    X = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / sr)
    return float((X[:, f < fc].sum(-1) / X.sum(-1)).max())


def paired(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    d = (a - b.reindex(a.index)).dropna()
    rng = np.random.default_rng(seed)
    boots = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "median": d.median(), "frac_better": (d > 0).mean(), "p_wilcoxon": wilcoxon(d).pvalue,
            "n_worse_2db": int((d < -2).sum()), "n_better_2db": int((d > 2).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--rumble_audio", default=None, help="zero-shot audio dir of the rumble example")
    a = ap.parse_args()
    R = Path(a.results); out = R / "report" / "extra"; out.mkdir(parents=True, exist_ok=True)
    S = {k: pd.read_csv(R / k / "per_utt.csv").set_index("id") for k in ["zeroshot", "ft_full", "blstm"]
         if (R / k / "per_utt.csv").exists()}
    z, f = S["zeroshot"], S["ft_full"]
    md = ["# Extra analysis", ""]

    # 1. paired comparison
    p = paired(f.si_sdri, z.si_sdri)
    md += ["## Fine-tuned vs zero-shot (paired over the same test mixtures)", "",
           f"- Mean change {p['mean']:+.2f} dB, 95% bootstrap CI [{p['ci_lo']:+.2f}, {p['ci_hi']:+.2f}], "
           f"median {p['median']:+.2f} dB",
           f"- Fine-tuned better on {100*p['frac_better']:.1f}% of mixtures; Wilcoxon signed-rank p = {p['p_wilcoxon']:.1e}",
           f"- {p['n_better_2db']} mixtures gained more than 2 dB, {p['n_worse_2db']} lost more than 2 dB",
           f"- STOI change {(f.stoi - z.stoi).mean():+.4f}, PESQ change {(f.pesq - z.pesq).mean():+.3f}", ""]
    t = pd.DataFrame({k: d.groupby("sex_pair").si_sdri.mean() for k, d in S.items()})
    t["ft_gain"] = t.ft_full - t.zeroshot
    t["n"] = z.groupby("sex_pair").size()
    md += ["## Mean SI-SDRi by speaker-sex pair", "", t.round(2).to_markdown(), ""]

    # 2. regressions
    d = f.si_sdri - z.si_sdri
    worse = d < -2
    md += ["## Regressions (fine-tuned more than 2 dB worse than zero-shot)", "",
           f"- n = {worse.sum()}; same-sex share {100*f.sex_pair[worse].isin(['FF','MM']).mean():.0f}% "
           f"vs {100*f.sex_pair.isin(['FF','MM']).mean():.0f}% overall",
           f"- mean swap fraction: fine-tuned {f.swap_frac[worse].mean():.2f}, zero-shot {z.swap_frac[worse].mean():.2f}",
           f"- mean STOI: fine-tuned {f.stoi[worse].mean():.3f}, zero-shot {z.stoi[worse].mean():.3f}", "",
           pd.DataFrame({"zero-shot": z.si_sdri[worse], "fine-tuned": f.si_sdri[worse], "sex pair": f.sex_pair[worse],
                         "dF0 (Hz)": f.df0[worse]}).sort_values("fine-tuned").round(2).to_markdown(), ""]

    # 3. low-frequency energy in the references (LibriSpeech recording rumble)
    info = find_libri2mix(a.data_root)["test"]
    lf = pd.Series({u: low_freq_share(load_triplet(info, u)[1].numpy()) for u in z.index}, name="low_freq_share")
    lf.to_csv(out / "low_freq_share.csv")
    g = lf > 0.2
    md += ["## Low-frequency (<100 Hz) energy in the reference sources", "",
           f"- {g.sum()} of {len(lf)} mixtures have a reference with more than 20% of its energy below 100 Hz "
           f"({(lf > 0.5).sum()} above 50%)"]
    for k, dd in S.items():
        rho = spearmanr(dd.si_sdri, lf.reindex(dd.index)).correlation
        md.append(f"- {k}: SI-SDRi {dd.si_sdri[g].mean():.2f} dB with rumble vs {dd.si_sdri[~g].mean():.2f} dB without; "
                  f"Spearman with low-frequency share {rho:+.2f}")
    md += [f"- fine-tuning gain: {d[g].mean():+.2f} dB with rumble vs {d[~g].mean():+.2f} dB without", ""]

    # 4. what predicts difficulty
    rows = {}
    for k, dd in S.items():
        x = dd.assign(level_diff=dd.src_level_diff_db.abs(), low_freq=lf.reindex(dd.index))
        rows[k] = {c: spearmanr(x.si_sdri, x[c], nan_policy="omit").correlation
                   for c in ["df0", "overlap", "level_diff", "low_freq", "swap_frac", "dur_s"]}
    md += ["## Spearman correlation of SI-SDRi with each diagnostic", "", pd.DataFrame(rows).T.round(2).to_markdown(), ""]

    # 5. training summaries
    for run in ["ft_full", "blstm"]:
        h = R.parent / "runs" / run / "history.csv"
        if h.exists():
            hh = pd.read_csv(h)
            md.append(f"- {run}: {len(hh)-1} epochs, {int(hh.step.iloc[-1])} optimiser steps, {hh.minutes.iloc[-1]:.0f} min, "
                      f"dev SI-SDRi {hh.val_si_sdri.iloc[0]:.2f} -> {hh.val_si_sdri.iloc[-1]:.2f} dB (best {hh.val_si_sdri.max():.2f})")
    (out / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # figure: SI-SDRi vs low-frequency share
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for j, k in enumerate(["zeroshot", "ft_full"]):
        ax.scatter(lf.reindex(S[k].index).clip(lower=1e-4) * 100, S[k].si_sdri, s=10, lw=0, alpha=0.6,
                   color=SERIES[j], label=k)
    ax.set_xscale("log"); ax.axvline(20, color=INK2, lw=1, ls="--")
    ax.set_xlabel("largest share of reference energy below 100 Hz (%)"); ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Low-frequency rumble in the references vs separation score", loc="left", fontsize=10)
    ax.legend(loc="lower left")
    fig.savefig(out / "fig_low_freq.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # figure: spectra of the rumble example (zero-shot)
    if a.rumble_audio and Path(a.rumble_audio).exists():
        ad = Path(a.rumble_audio)
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
        for i, ax in enumerate(axes):
            for j, (kind, ls) in enumerate([("ref", "-"), ("est", "--")]):
                x, sr = sf.read(ad / f"s{i+1}_{kind}.wav")
                X = np.abs(np.fft.rfft(x - x.mean())) ** 2; fr = np.fft.rfftfreq(len(x), 1 / sr)
                k_ = max(1, len(X) // 400)
                Xs = np.convolve(X, np.ones(k_) / k_, mode="same")
                ax.semilogy(fr, Xs + 1e-12, ls, color=SERIES[j], lw=1.5, label=f"{kind}")
            ax.set_xlim(0, 1000); ax.set_title(f"speaker {i+1}", loc="left", fontsize=9)
            ax.axvline(100, color=INK2, lw=1, ls=":"); ax.set_xlabel("frequency (Hz)")
        axes[0].set_ylabel("power"); axes[0].legend(loc="upper right")
        fig.suptitle("Rumble example: the <100 Hz energy of speaker 2's reference ends up in output 1",
                     x=0.01, y=1.06, ha="left", fontsize=9)
        fig.savefig(out / "fig_rumble_example.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print((out / "stats.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
