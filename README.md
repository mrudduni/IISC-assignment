# IISC assignment: audio-video AI / generative audio

| Part | Where |
|---|---|
| Report (all three parts, AI-use statement) | [`Report.pdf`](Report.pdf) |
| 1. Two-speaker source separation: code | [`part1_separation/`](part1_separation/) |
| 1. Results: per-mixture metrics, figures, audio examples, training logs | [`part1_results/`](part1_results/) |
| 2–3. Evaluating video-to-audio generation, failure analysis | written answers, Sections 2–3 of [`Report.pdf`](Report.pdf) |

## Part 1: fine-tuning SR-CorrNet on Libri2Mix

**Question.** How well does a state-of-the-art separator trained on WSJ0-2mix transfer to Libri2Mix, and how much of
the gap does fine-tuning close compared with training a simple model from scratch on the same data?

**Systems** (evaluated on the same 500 random Libri2Mix `test` mixtures, `mix_clean`, 8 kHz):

| name | description |
|---|---|
| `zeroshot` | official SR-CorrNet checkpoint `shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk`, no training |
| `ft_full` | that checkpoint fine-tuned end-to-end on Libri2Mix `train-100` |
| `blstm` | own 3-layer BLSTM mask network trained from scratch |

### What is mine and what is not

| Component | Source |
|---|---|
| SR-CorrNet network (`Model`), its STFT/iSTFT layers, pretrained weights | [dmlguq456/SR_CorrNet_SS](https://github.com/dmlguq456/SR_CorrNet_SS) (MIT), [arXiv:2603.29097](https://arxiv.org/abs/2603.29097) |
| Libri2Mix discovery, on-the-fly mixing from LibriMix metadata, resampling, crops (`common.py`) | own; metadata CSVs from [JorisCos/LibriMix](https://github.com/JorisCos/LibriMix) |
| Waveform wrapper around SR-CorrNet, checkpoint loading, chunked long-file inference with permutation stitching | own |
| PIT SI-SDR loss, auxiliary magnitude loss (re-implemented after the repo's) | own |
| Training loop with budget-sized cosine schedule, resume, encoder freezing (`train.py`) | own |
| BLSTM mask baseline | own |
| Evaluation: SI-SDRi, windowed swap rate, pitch-gap / level / overlap / speaker-sex diagnostics (`evaluate.py`) | own; SDR from `mir_eval`, STOI from `pystoi`, PESQ from `pesq` |
| Analysis, tables, figures (`analyze.py`) | own |

### Setup

Kaggle (recommended, no local GPU needed): upload `part1_separation/kaggle_part1.ipynb`, enable GPU and Internet,
and run all. No dataset upload is needed: the notebook downloads LibriSpeech and the LibriMix metadata itself.

Local / any GPU machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd part1_separation
```

**Data.** `--data_root` takes one or more comma-separated folders and accepts either form of Libri2Mix:

- **LibriSpeech + LibriMix metadata (what the notebook uses).** Point it at folders containing `train-clean-100`,
  `dev-clean` and `test-clean` from LibriSpeech, and at `metadata/` from the
  [LibriMix repo](https://github.com/JorisCos/LibriMix). Mixtures are built on the fly exactly like the official
  generator's `min` / `mix_clean` at 8 kHz: each source times its metadata gain, resampled with the same polyphase
  filter, truncated to the shorter source, and summed. This saves about 14 GB of generated WAVs and adds the
  speakers' sex from the metadata.
- **Generated Libri2Mix WAVs.** Split folders `train-100`, `dev`, `test` that contain `mix_clean/`, `s1/`, `s2/`, at
  `wav8k` or `wav16k`.

```bash
python common.py --data_root /path/to/LibriMix
```

### Training

```bash
# fine-tune with gradient checkpointing, LR schedule sized to a 1.25 h budget
python train.py --data_root DATA --model srcorrnet --init pretrained --run_name ft_full --budget_h 1.25 --grad_ckpt --batch_size 1 --accum 4
# optional ablation (implemented, not run for the report): frozen encoder
python train.py --data_root DATA --model srcorrnet --init pretrained --freeze_encoder --run_name ft_frozen --budget_h 0.5 --grad_ckpt --batch_size 1 --accum 4
# from-scratch baseline
python train.py --data_root DATA --model baseline --init scratch --lr 1e-3 --batch_size 8 --accum 1 --run_name blstm --budget_h 0.5
# continue an interrupted run
python train.py ... --run_name ft_full --resume
```

Outputs go to `runs/<name>/`: `best.pt` (best dev SI-SDRi; epoch 0 is the starting point), `last.pt`, `history.csv`.
Without checkpointing, SR-CorrNet training stores about 4.2 GiB of activations per second of audio per sample,
so one 4 s clip (about 17 GiB) overflows a 16 GB GPU. Add `--grad_ckpt --batch_size 1 --accum 4`: gradient
checkpointing recomputes each block in the backward pass, gives identical gradients, and stores about 0.6 GiB.

### Evaluation

```bash
python evaluate.py --data_root DATA --model srcorrnet --ckpt pretrained --out results/zeroshot --sdr --perceptual --max_utts 500
python evaluate.py --data_root DATA --model srcorrnet --ckpt runs/ft_full/best.pt --out results/ft_full --sdr --perceptual --max_utts 500
python analyze.py select --runs zeroshot=results/zeroshot ft_full=results/ft_full --main ft_full --ref zeroshot
python evaluate.py --data_root DATA --ckpt runs/ft_full/best.pt --out results/ft_full --ids_file results/report/example_ids.txt --save_audio
python analyze.py report --runs zeroshot=results/zeroshot ft_full=results/ft_full \
    --histories ft_full=runs/ft_full/history.csv --main ft_full --ref zeroshot
```

`--max_utts 500` evaluates a fixed random subset (seed 0), so every system is scored on the same mixtures.
`results/report/results.md` holds the summary table, stratified tables, and figures. `results/<system>/audio/`
holds the mixture, references, and estimates for each selected example.

Extra statistics used in the report (paired bootstrap CI, Wilcoxon test, regressions, low-frequency rumble
analysis) run on CPU and need the LibriSpeech test audio:

```bash
python extra_analysis.py --results ../part1_results/results --data_root /path/to/LibriMix --rumble_audio ../part1_results/results/report/extra/rumble_example_zeroshot
```

### Results (500 random test mixtures, 8 kHz)

| System | SI-SDRi (dB) | STOI | PESQ-NB |
|---|---|---|---|
| Mixture | 0.00 | 0.712 | 1.61 |
| Zero-shot SR-CorrNet | 18.87 ± 3.91 | 0.972 | 3.75 |
| Fine-tuned SR-CorrNet (1 epoch) | 20.05 ± 3.69 | 0.973 | 3.85 |
| BLSTM from scratch | 6.29 ± 4.11 | – | – |

Fine-tuning gains +1.18 dB per mixture (95% CI +1.01 to +1.35). Full results are in `part1_results/results/report/`,
with analysis and failure cases in [`Report.pdf`](Report.pdf). The BLSTM's STOI/PESQ were not computed.

### Inference

**Fine-tuned checkpoint:** [Google Drive](https://drive.google.com/file/d/1H9JNc70oWLdpq6qOgkrpWNJ-c8YXvT8N/view?usp=sharing)
(`best.pt`, 54 MB, SR-CorrNet after one epoch on Libri2Mix `train-100`). Save it as
`part1_separation/runs/ft_full/best.pt`. The zero-shot model needs no download; `--ckpt pretrained` fetches it from
Hugging Face.

```bash
python inference.py --ckpt pretrained --input mixture.wav --output separated/          # zero-shot SR-CorrNet
python inference.py --ckpt runs/ft_full/best.pt --input mixture.wav --output separated/   # fine-tuned
```

`--input` can be a single file or a folder of WAVs; any sample rate is resampled to 8 kHz.

### Tested

All scripts were smoke-tested on CPU. On a tiny Libri2Mix-shaped set built from three LibriSpeech clips, fine-tuning,
resume, frozen-encoder training, the baseline with the budget planner, example export, the report, and inference all
ran end to end. On the real data, on-the-fly mixing found 13,900 / 3,000 / 3,000 train / dev / test mixtures.
The reported runs used one T4 (16 GB) on Kaggle: fine-tuning took 40 min (one epoch, 150 updates, 4 s crops with
gradient checkpointing) and the BLSTM 29 min.
