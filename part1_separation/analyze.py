"""Compare evaluated systems and produce report tables / figures (own implementation).

  # 1) choose qualitative + failure examples from the per-utterance CSVs
  python analyze.py select --runs zeroshot=results/zeroshot ft_full=results/ft_full \
      --main ft_full --ref zeroshot --out results/report
  # 2) after re-running evaluate.py --ids_file ... --save_audio, build tables and figures
  python analyze.py report --runs zeroshot=results/zeroshot ft_full=results/ft_full \
      --histories ft_full=runs/ft_full/history.csv --main ft_full --ref zeroshot --out results/report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

# Fixed categorical order (validated default palette); colour follows the system, not its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def kv(items):
    return dict(i.split("=", 1) for i in items or [])


def load_runs(runs: dict) -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(Path(v) / "per_utt.csv") for k, v in runs.items()}


# Example selection
def select(a):
    dfs = load_runs(kv(a.runs))
    m = dfs[a.main].set_index("id")
    picks = []
    def add(i, why):
        if i not in [p[0] for p in picks]:
            picks.append((i, why))
    s = m.si_sdri.sort_values()
    add(s.index[-1], "best")
    add(s.index[len(s) // 2], "median")
    add(s.index[0], "worst")
    add(s.index[1], "2nd worst")
    sw = m[m.swap_frac.notna()].swap_frac.sort_values()
    if len(sw) and sw.iloc[-1] > 0:
        add(sw.index[-1], "highest mid-utterance swap rate")
    add(m.df0.idxmin(), "smallest pitch gap between speakers")
    add(m.src_level_diff_db.abs().idxmax(), "largest level difference between speakers")
    if a.ref and a.ref in dfs:
        r = dfs[a.ref].set_index("id")
        gain = (m.si_sdri - r.si_sdri.reindex(m.index)).dropna().sort_values()
        add(gain.index[-1], f"largest gain over {a.ref}")
        add(gain.index[0], f"largest loss vs {a.ref}")
    if "sex_pair" in m:
        same = m[m.sex_pair.isin(["FF", "MM"])]
        if len(same):
            add(same.si_sdri.idxmin(), "worst same-sex mixture")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "example_ids.txt").write_text("\n".join(i for i, _ in picks) + "\n")
    pd.DataFrame(picks, columns=["id", "reason"]).merge(
        m.reset_index(), on="id").to_csv(out / "example_ids.csv", index=False)
    for i, why in picks:
        print(f"{why:45s} {i}  SI-SDRi={m.loc[i, 'si_sdri']:.2f}")


# Report: tables and figures
def fmt(x, d=2):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def summary_table(dfs) -> str:
    cols = [("si_sdri", "SI-SDRi (dB)"), ("sdri", "SDRi (dB)"), ("stoi", "STOI"), ("pesq", "PESQ-NB")]
    lines = ["| System | N | " + " | ".join(c[1] for c in cols) + " | median SI-SDRi | % utts < 5 dB | % utts with a swap |",
             "|---|---|" + "---|" * len(cols) + "---|---|---|"]
    first = next(iter(dfs.values()))
    if "stoi_in" in first:
        lines.append(f"| Mixture (input) | {len(first)} | 0.00 | 0.00 | {fmt(first.stoi_in.mean(),3)} | "
                     f"{fmt(first.pesq_in.mean())} | – | – | – |")
    for k, d in dfs.items():
        cells = []
        for c, _ in cols:
            if c in d:
                cells.append(f"{fmt(d[c].mean(), 3 if c=='stoi' else 2)} ± {fmt(d[c].std(), 3 if c=='stoi' else 2)}")
            else:
                cells.append("–")
        lines.append(f"| {k} | {len(d)} | " + " | ".join(cells) +
                     f" | {fmt(d.si_sdri.median())} | {100*(d.si_sdri<5).mean():.1f} | {100*(d.swap_frac>0).mean():.1f} |")
    return "\n".join(lines)


def stratified_table(dfs, col, bins, labels, title) -> tuple[str, pd.DataFrame]:
    rows = {}
    for k, d in dfs.items():
        b = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
        rows[k] = d.groupby(b, observed=False).si_sdri.mean()
    counts = pd.cut(next(iter(dfs.values()))[col], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels)
    t = pd.DataFrame(rows)
    lines = [f"**{title}**", "", "| bin | N | " + " | ".join(t.columns) + " |", "|---|---|" + "---|" * len(t.columns)]
    for lab in labels:
        lines.append(f"| {lab} | {counts[lab]} | " + " | ".join(fmt(t.loc[lab, c]) for c in t.columns) + " |")
    return "\n".join(lines), t


def grouped_bars(t: pd.DataFrame, xlabel: str, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    n = len(t.columns); w = 0.8 / n; x = np.arange(len(t.index))
    for j, c in enumerate(t.columns):
        ax.bar(x + (j - (n - 1) / 2) * w, t[c].values, w * 0.92, color=SERIES[j], label=c, zorder=3)
    ax.set_xticks(x, t.index.astype(str)); ax.set_xlabel(xlabel); ax.set_ylabel("mean SI-SDRi (dB)")
    ax.grid(axis="x", visible=False); ax.set_title(title, loc="left", fontsize=10)
    ax.legend(ncol=n, loc="upper left", bbox_to_anchor=(0, -0.22))
    fig.savefig(path); plt.close(fig)


def spectrogram_grid(audio_dir: Path, path: Path, title: str):
    names = ["mix", "s1_ref", "s1_est", "s2_ref", "s2_est"]
    files = [audio_dir / f"{n}.wav" for n in names]
    if not all(f.exists() for f in files):
        return False
    fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 7.0), sharex=True)
    for ax, n, f in zip(axes, names, files):
        x, sr = sf.read(f)
        ax.specgram(x + 1e-9, NFFT=256, Fs=sr, noverlap=192, cmap="magma", vmin=-120, vmax=-20)
        ax.set_ylabel(n, rotation=0, ha="right", va="center"); ax.grid(False)
        ax.set_yticks([0, 2000, 4000]); ax.set_yticklabels(["0", "2k", "4k"])
    axes[-1].set_xlabel("time (s)"); axes[0].set_title(title, loc="left", fontsize=9)
    fig.savefig(path); plt.close(fig)
    return True


def report(a):
    runs = kv(a.runs)
    dfs = load_runs(runs)
    out = Path(a.out); figs = out / "figs"; figs.mkdir(parents=True, exist_ok=True)
    md = ["# Part 1 results", "", "All metrics on full-length test utterances at 8 kHz, utterance-level PIT.", "",
          summary_table(dfs), ""]

    # distribution of SI-SDRi
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    lo = min(d.si_sdri.min() for d in dfs.values()); hi = max(d.si_sdri.max() for d in dfs.values())
    bins = np.linspace(np.floor(lo), np.ceil(hi), 50)
    for j, (k, d) in enumerate(dfs.items()):
        ax.hist(d.si_sdri, bins=bins, histtype="step", lw=2, color=SERIES[j], label=f"{k} (mean {d.si_sdri.mean():.1f})")
    ax.set_xlabel("SI-SDRi per utterance (dB)"); ax.set_ylabel("utterances"); ax.legend(loc="upper left")
    ax.set_title("Distribution of per-utterance SI-SDR improvement", loc="left", fontsize=10)
    fig.savefig(figs / "hist_si_sdri.png"); plt.close(fig)
    md += ["![](figs/hist_si_sdri.png)", ""]

    # stratification
    for col, bins_, labels, xl, fname, title in [
        ("df0", [0, 20, 50, 100, 1e4], ["<20 Hz", "20–50", "50–100", ">100 Hz"],
         "|median F0(s1) − median F0(s2)|", "by_df0.png", "SI-SDRi vs pitch gap between speakers"),
        ("src_level_diff_db", None, ["0–1 dB", "1–2", "2–3", "3–5", ">5 dB"],
         "|level difference between sources|", "by_level.png", "SI-SDRi vs source level difference"),
        ("dur_s", [0, 4, 6, 8, 12, 1e4], ["<4 s", "4–6", "6–8", "8–12", ">12 s"],
         "utterance duration", "by_dur.png", "SI-SDRi vs utterance duration"),
    ]:
        if col == "src_level_diff_db":
            dd = {k: d.assign(src_level_diff_db=d.src_level_diff_db.abs()) for k, d in dfs.items()}
            bins_ = [0, 1, 2, 3, 5, 1e4]
        else:
            dd = dfs
        txt, t = stratified_table(dd, col, bins_, labels, title)
        grouped_bars(t, xl, figs / fname, title)
        md += [txt, "", f"![](figs/{fname})", ""]

    # speaker-sex pairing (available when the data comes from LibriMix metadata)
    if all("sex_pair" in d for d in dfs.values()):
        t = pd.DataFrame({k: d.groupby("sex_pair").si_sdri.mean() for k, d in dfs.items()})
        t = t.reindex([x for x in ["FM", "FF", "MM"] if x in t.index])
        counts = next(iter(dfs.values())).sex_pair.value_counts()
        lines = ["**SI-SDRi by speaker-sex pairing**", "", "| pair | N | " + " | ".join(t.columns) + " |",
                 "|---|---|" + "---|" * len(t.columns)]
        for lab in t.index:
            lines.append(f"| {lab} | {counts.get(lab, 0)} | " + " | ".join(fmt(t.loc[lab, c]) for c in t.columns) + " |")
        grouped_bars(t, "speaker-sex pair (F = female, M = male)", figs / "by_sex_pair.png",
                     "SI-SDRi for mixed-sex vs same-sex mixtures")
        md += ["\n".join(lines), "", "![](figs/by_sex_pair.png)", ""]

    # per-utterance before/after scatter
    if a.ref and a.ref in dfs and a.main in dfs:
        m = dfs[a.main].set_index("id").si_sdri; r = dfs[a.ref].set_index("id").si_sdri.reindex(m.index)
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(r, m, s=9, color=SERIES[0], alpha=0.5, lw=0)
        lim = [min(r.min(), m.min()) - 1, max(r.max(), m.max()) + 1]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{a.ref} SI-SDRi (dB)"); ax.set_ylabel(f"{a.main} SI-SDRi (dB)")
        ax.set_title("Per-utterance change", loc="left", fontsize=10)
        fig.savefig(figs / "scatter_main_vs_ref.png"); plt.close(fig)
        better = (m > r).mean()
        md += [f"{a.main} beats {a.ref} on {100*better:.1f}% of utterances "
               f"(mean change {np.nanmean(m - r):+.2f} dB).", "", "![](figs/scatter_main_vs_ref.png)", ""]

    # learning curves
    hist = kv(a.histories)
    if hist:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        names = list(dfs.keys())
        for k, p in hist.items():
            h = pd.read_csv(p)
            j = names.index(k) if k in names else len(names) + list(hist).index(k)
            ax.plot(h.epoch, h.val_si_sdri, marker="o", ms=4, color=SERIES[j % len(SERIES)], label=k)
        ax.set_xlabel("epoch (0 = before training)"); ax.set_ylabel("dev SI-SDRi (dB)"); ax.legend()
        ax.set_title("Validation SI-SDRi during training", loc="left", fontsize=10)
        fig.savefig(figs / "learning_curves.png"); plt.close(fig)
        md += ["![](figs/learning_curves.png)", ""]

    # worst utterances with diagnostics
    if a.main in dfs:
        w = dfs[a.main].nsmallest(10, "si_sdri")
        md += [f"**10 worst utterances for {a.main}**", "",
               "| id | SI-SDRi | s1 | s2 | ΔF0 (Hz) | level diff (dB) | overlap | swap frac | dur (s) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for _, r_ in w.iterrows():
            md.append(f"| {r_.id} | {fmt(r_.si_sdri)} | {fmt(r_.si_sdr_s1)} | {fmt(r_.si_sdr_s2)} | {fmt(r_.df0,0)} | "
                      f"{fmt(r_.src_level_diff_db,1)} | {fmt(r_.overlap)} | {fmt(r_.swap_frac)} | {fmt(r_.dur_s,1)} |")
        md.append("")
        corr = dfs[a.main][["si_sdri", "df0", "src_level_diff_db", "overlap", "swap_frac", "dur_s"]].copy()
        corr["src_level_diff_db"] = corr.src_level_diff_db.abs()
        c = corr.corr(method="spearman")["si_sdri"].drop("si_sdri")
        md += ["Spearman correlation of SI-SDRi with each diagnostic (" + a.main + "): " +
               ", ".join(f"{k} {v:+.2f}" for k, v in c.items()), ""]

    # spectrograms of the selected examples
    ex = out / "example_ids.csv"
    if ex.exists():
        md += ["## Qualitative examples", ""]
        for _, e in pd.read_csv(ex).iterrows():
            stem = Path(e.id).stem
            for k in dfs:
                ad = Path(runs[k]) / "audio" / stem
                f = figs / f"spec_{k}_{stem}.png"
                si = dfs[k].set_index("id").si_sdri.get(e.id, np.nan)
                if spectrogram_grid(ad, f, f"{k} | {e.reason} | SI-SDRi {si:.1f} dB"):
                    md += [f"*{e.reason}* ({k}, `{e.id}`): ![]({f.relative_to(out).as_posix()})", ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out/'results.md'} and {len(list(figs.glob('*.png')))} figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "report"])
    ap.add_argument("--runs", nargs="+", required=True, help="name=results_dir ...")
    ap.add_argument("--histories", nargs="*", default=[], help="name=history.csv ...")
    ap.add_argument("--main", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="results/report")
    a = ap.parse_args()
    select(a) if a.cmd == "select" else report(a)


if __name__ == "__main__":
    main()
