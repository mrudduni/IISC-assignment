# Extra analysis

## Fine-tuned vs zero-shot (paired over the same test mixtures)

- Mean change +1.18 dB, 95% bootstrap CI [+1.01, +1.35], median +1.02 dB
- Fine-tuned better on 93.8% of mixtures; Wilcoxon signed-rank p = 5.6e-64
- 80 mixtures gained more than 2 dB, 13 lost more than 2 dB
- STOI change +0.0011, PESQ change +0.106

## Mean SI-SDRi by speaker-sex pair

| sex_pair   |   zeroshot |   ft_full |   blstm |   ft_gain |   n |
|:-----------|-----------:|----------:|--------:|----------:|----:|
| FF         |      18.65 |     19.72 |    3.37 |      1.07 | 134 |
| FM         |      19.55 |     20.72 |    8.89 |      1.17 | 249 |
| MM         |      17.68 |     19.01 |    4.1  |      1.33 | 117 |

## Regressions (fine-tuned more than 2 dB worse than zero-shot)

- n = 13; same-sex share 85% vs 50% overall
- mean swap fraction: fine-tuned 0.02, zero-shot 0.00
- mean STOI: fine-tuned 0.947, zero-shot 0.967

| id                                |   zero-shot |   fine-tuned | sex pair   |   dF0 (Hz) |
|:----------------------------------|------------:|-------------:|:-----------|-----------:|
| 1580-141083-0034_3729-6852-0024   |       11.77 |         5.14 | FF         |       9.11 |
| 6829-68771-0025_4970-29095-0022   |        8.13 |         5.81 | FF         |      62.26 |
| 4970-29095-0008_5142-36377-0007   |       21.12 |         7.43 | FF         |      19.37 |
| 2830-3980-0018_5105-28233-0003    |       17.06 |         7.56 | MM         |      58.19 |
| 1580-141083-0041_5142-36377-0024  |       16.67 |         8.12 | FF         |      29.83 |
| 3575-170457-0030_237-134493-0015  |       10.82 |         8.17 | FF         |       6.97 |
| 1320-122612-0001_2300-131720-0002 |       16.7  |        11.17 | MM         |       8.6  |
| 7729-102255-0025_2830-3980-0029   |       17.12 |        13.87 | MM         |      32.02 |
| 2830-3980-0032_4507-16021-0022    |       19.05 |        14.88 | FM         |      79.19 |
| 1580-141084-0002_4992-41797-0004  |       19.7  |        15.13 | FF         |      28.16 |
| 1089-134691-0014_2830-3980-0041   |       20.79 |        15.77 | MM         |      31.22 |
| 4446-2273-0022_2830-3980-0023     |       19.65 |        17.56 | FM         |      67.95 |
| 1995-1837-0021_4970-29095-0026    |       22.12 |        19.52 | FF         |      18    |

## Low-frequency (<100 Hz) energy in the reference sources

- 11 of 500 mixtures have a reference with more than 20% of its energy below 100 Hz (2 above 50%)
- zeroshot: SI-SDRi 11.76 dB with rumble vs 19.03 dB without; Spearman with low-frequency share -0.39
- ft_full: SI-SDRi 15.54 dB with rumble vs 20.15 dB without; Spearman with low-frequency share -0.33
- blstm: SI-SDRi 5.64 dB with rumble vs 6.30 dB without; Spearman with low-frequency share +0.11
- fine-tuning gain: +3.79 dB with rumble vs +1.12 dB without

## Spearman correlation of SI-SDRi with each diagnostic

|          |   df0 |   overlap |   level_diff |   low_freq |   swap_frac |   dur_s |
|:---------|------:|----------:|-------------:|-----------:|------------:|--------:|
| zeroshot |  0.22 |     -0.33 |        -0.07 |      -0.39 |       -0.15 |    0.06 |
| ft_full  |  0.22 |     -0.35 |        -0.07 |      -0.33 |       -0.13 |    0.09 |
| blstm    |  0.59 |     -0.12 |         0.04 |       0.11 |       -0.63 |    0.07 |

- ft_full: 1 epochs, 150 optimiser steps, 40 min, dev SI-SDRi 18.68 -> 20.19 dB (best 20.19)
- blstm: 100 epochs, 15000 optimiser steps, 29 min, dev SI-SDRi 0.01 -> 6.75 dB (best 6.76)
