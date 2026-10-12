# Part 1 results

All metrics on full-length test utterances at 8 kHz, utterance-level PIT.

| System | N | SI-SDRi (dB) | SDRi (dB) | STOI | PESQ-NB | median SI-SDRi | % utts < 5 dB | % utts with a swap |
|---|---|---|---|---|---|---|---|---|
| Mixture (input) | 500 | 0.00 | 0.00 | 0.712 | 1.61 | – | – | – |
| zeroshot | 500 | 18.87 ± 3.91 | – | 0.972 ± 0.026 | 3.75 ± 0.30 | 19.60 | 1.4 | 0.8 |
| ft_full | 500 | 20.05 ± 3.69 | – | 0.973 ± 0.026 | 3.85 ± 0.27 | 20.74 | 0.6 | 0.6 |
| blstm | 500 | 6.29 ± 4.11 | – | – | – | 6.81 | 37.4 | 17.6 |

![](figs/hist_si_sdri.png)

**SI-SDRi vs pitch gap between speakers**

| bin | N | zeroshot | ft_full | blstm |
|---|---|---|---|---|
| <20 Hz | 111 | 17.59 | 18.89 | 3.26 |
| 20–50 | 148 | 18.59 | 19.79 | 4.78 |
| 50–100 | 180 | 19.57 | 20.61 | 8.35 |
| >100 Hz | 61 | 19.84 | 21.16 | 9.38 |

![](figs/by_df0.png)

**SI-SDRi vs source level difference**

| bin | N | zeroshot | ft_full | blstm |
|---|---|---|---|---|
| 0–1 dB | 108 | 18.95 | 19.96 | 6.19 |
| 1–2 | 91 | 19.15 | 20.33 | 6.33 |
| 2–3 | 89 | 19.19 | 20.48 | 6.10 |
| 3–5 | 125 | 18.91 | 20.26 | 6.42 |
| >5 dB | 87 | 18.10 | 19.14 | 6.38 |

![](figs/by_level.png)

**SI-SDRi vs utterance duration**

| bin | N | zeroshot | ft_full | blstm |
|---|---|---|---|---|
| <4 s | 187 | 18.63 | 19.81 | 5.76 |
| 4–6 | 199 | 19.11 | 20.21 | 6.89 |
| 6–8 | 79 | 18.95 | 20.09 | 5.80 |
| 8–12 | 34 | 18.69 | 20.42 | 6.76 |
| >12 s | 1 | 18.18 | 19.04 | 8.27 |

![](figs/by_dur.png)

**SI-SDRi by speaker-sex pairing**

| pair | N | zeroshot | ft_full | blstm |
|---|---|---|---|---|
| FM | 249 | 19.55 | 20.72 | 8.89 |
| FF | 134 | 18.65 | 19.72 | 3.37 |
| MM | 117 | 17.68 | 19.01 | 4.10 |

![](figs/by_sex_pair.png)

ft_full beats zeroshot on 93.8% of utterances (mean change +1.18 dB).

![](figs/scatter_main_vs_ref.png)

![](figs/learning_curves.png)

**10 worst utterances for ft_full**

| id | SI-SDRi | s1 | s2 | ΔF0 (Hz) | level diff (dB) | overlap | swap frac | dur (s) |
|---|---|---|---|---|---|---|---|---|
| 3729-6852-0031_121-127105-0034 | -3.94 | -3.44 | -4.33 | 0 | 3.0 | 0.45 | 0.25 | 4.4 |
| 5142-36377-0001_5683-32866-0012 | 0.11 | -0.87 | 0.98 | 100 | -1.5 | 0.66 | 0.20 | 5.4 |
| 2830-3980-0038_121-127105-0007 | 4.99 | 4.01 | 5.96 | 97 | -1.7 | 0.73 | 0.00 | 5.5 |
| 1580-141083-0034_3729-6852-0024 | 5.14 | 8.47 | 1.94 | 9 | 5.2 | 0.66 | 0.00 | 5.5 |
| 6829-68771-0025_4970-29095-0022 | 5.81 | 5.41 | 6.20 | 62 | -0.2 | 0.45 | 0.25 | 4.8 |
| 3575-170457-0051_121-121726-0008 | 6.19 | 10.69 | 1.53 | 8 | 7.3 | 0.44 | 0.00 | 4.9 |
| 4992-23283-0005_6829-68769-0034 | 7.06 | 8.37 | 5.96 | 12 | 2.8 | 0.59 | 0.00 | 6.1 |
| 4970-29095-0008_5142-36377-0007 | 7.43 | 9.92 | 4.91 | 19 | 3.0 | 0.43 | 0.00 | 3.0 |
| 2830-3980-0018_5105-28233-0003 | 7.56 | 9.62 | 5.45 | 58 | 5.3 | 0.74 | 0.00 | 3.4 |
| 1580-141084-0024_2094-142345-0025 | 7.83 | 5.35 | 10.22 | 33 | -0.7 | 0.42 | 0.00 | 3.6 |

Spearman correlation of SI-SDRi with each diagnostic (ft_full): df0 +0.22, src_level_diff_db -0.07, overlap -0.35, swap_frac -0.13, dur_s +0.09

## Qualitative examples

*best* (zeroshot, `1995-1837-0019_1089-134686-0009`): ![](figs/spec_zeroshot_1995-1837-0019_1089-134686-0009.png)

*best* (ft_full, `1995-1837-0019_1089-134686-0009`): ![](figs/spec_ft_full_1995-1837-0019_1089-134686-0009.png)

*best* (blstm, `1995-1837-0019_1089-134686-0009`): ![](figs/spec_blstm_1995-1837-0019_1089-134686-0009.png)

*median* (zeroshot, `8230-279154-0030_121-127105-0001`): ![](figs/spec_zeroshot_8230-279154-0030_121-127105-0001.png)

*median* (ft_full, `8230-279154-0030_121-127105-0001`): ![](figs/spec_ft_full_8230-279154-0030_121-127105-0001.png)

*median* (blstm, `8230-279154-0030_121-127105-0001`): ![](figs/spec_blstm_8230-279154-0030_121-127105-0001.png)

*worst* (zeroshot, `3729-6852-0031_121-127105-0034`): ![](figs/spec_zeroshot_3729-6852-0031_121-127105-0034.png)

*worst* (ft_full, `3729-6852-0031_121-127105-0034`): ![](figs/spec_ft_full_3729-6852-0031_121-127105-0034.png)

*worst* (blstm, `3729-6852-0031_121-127105-0034`): ![](figs/spec_blstm_3729-6852-0031_121-127105-0034.png)

*2nd worst* (zeroshot, `5142-36377-0001_5683-32866-0012`): ![](figs/spec_zeroshot_5142-36377-0001_5683-32866-0012.png)

*2nd worst* (ft_full, `5142-36377-0001_5683-32866-0012`): ![](figs/spec_ft_full_5142-36377-0001_5683-32866-0012.png)

*2nd worst* (blstm, `5142-36377-0001_5683-32866-0012`): ![](figs/spec_blstm_5142-36377-0001_5683-32866-0012.png)

*largest level difference between speakers* (zeroshot, `5142-36586-0004_8230-279154-0040`): ![](figs/spec_zeroshot_5142-36586-0004_8230-279154-0040.png)

*largest level difference between speakers* (ft_full, `5142-36586-0004_8230-279154-0040`): ![](figs/spec_ft_full_5142-36586-0004_8230-279154-0040.png)

*largest level difference between speakers* (blstm, `5142-36586-0004_8230-279154-0040`): ![](figs/spec_blstm_5142-36586-0004_8230-279154-0040.png)

*largest gain over zeroshot* (zeroshot, `8555-284449-0012_1580-141084-0004`): ![](figs/spec_zeroshot_8555-284449-0012_1580-141084-0004.png)

*largest gain over zeroshot* (ft_full, `8555-284449-0012_1580-141084-0004`): ![](figs/spec_ft_full_8555-284449-0012_1580-141084-0004.png)

*largest gain over zeroshot* (blstm, `8555-284449-0012_1580-141084-0004`): ![](figs/spec_blstm_8555-284449-0012_1580-141084-0004.png)

*largest loss vs zeroshot* (zeroshot, `4970-29095-0008_5142-36377-0007`): ![](figs/spec_zeroshot_4970-29095-0008_5142-36377-0007.png)

*largest loss vs zeroshot* (ft_full, `4970-29095-0008_5142-36377-0007`): ![](figs/spec_ft_full_4970-29095-0008_5142-36377-0007.png)

*largest loss vs zeroshot* (blstm, `4970-29095-0008_5142-36377-0007`): ![](figs/spec_blstm_4970-29095-0008_5142-36377-0007.png)
