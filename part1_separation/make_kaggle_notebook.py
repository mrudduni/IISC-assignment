"""Build kaggle_part1.ipynb: a self-contained notebook that embeds the scripts in this folder
via %%writefile cells, so you upload ONE file to Kaggle. Re-run after editing any script:

    python make_kaggle_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRCORRNET_COMMIT = "7340365b9cc9a021bf7d400f52fce4b88593b67a"
LIBRIMIX_COMMIT = "b898ca375c03e4039cdb3dbe3dd485aad58ec7d1"
SCRIPTS = ["common.py", "train.py", "evaluate.py", "analyze.py", "inference.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Part 1: two-speaker separation on Libri2Mix, fine-tuning SR-CorrNet

**Before running:** Settings, Accelerator **GPU T4 x2 or P100**, Internet **On**. No dataset upload is needed:
the data cell downloads LibriSpeech (`train-clean-100`, `dev-clean`, `test-clean`) and the official LibriMix
metadata, and the scripts build Libri2Mix `min` / `mix_clean` mixtures on the fly at 8 kHz. If a LibriSpeech or
generated Libri2Mix dataset is attached under *Add Input*, it is found and used instead of downloading.

Systems compared on the full Libri2Mix test set (8 kHz, `mix_clean`):

| name | what |
|---|---|
| `zeroshot` | official SR-CorrNet checkpoint (trained on WSJ0-2mix), no training |
| `ft_full` | same checkpoint fine-tuned end-to-end on Libri2Mix train-100 |
| `ft_frozen` | fine-tuned with the separation encoder frozen (ablation) |
| `blstm` | own BLSTM mask baseline trained from scratch |

To fit Kaggle's 12 h session, use *Save Version, Save & Run All (Commit)* so it runs in the background, and set the
hour budgets in the next cell. Each training run saves `last.pt` every epoch; set `RESUME = True` to continue.
"""),
    code("""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less fragmentation on 16 GB GPUs
# Settings
LS_DIR, META_DIR = "/tmp/librispeech", "/tmp/LibriMix"   # download targets (not saved as output)
DATA_ROOT = f"/kaggle/input,{LS_DIR},{META_DIR}/metadata"   # all searched; comma-separated
QUICK = False                 # True = 10-minute pipeline check on tiny subsets
# Defaults fit about 3 h on a T4 (estimate). SR-CorrNet costs ~365 GFLOPs per second of audio,
# so the full 3000-utterance test set would take over 1 h per system; we use a fixed random 500.
FT_HOURS, FROZEN_HOURS, BLSTM_HOURS = 1.25, 0.5, 0.5   # wall-clock training budgets
RUN_FROZEN, RUN_BLSTM = False, True   # frozen-encoder ablation adds ~50 min
EVAL_EXTRA = "--perceptual"  # STOI + PESQ; add "--sdr" for BSS-eval SDR (slow)
RESUME = False
# Without gradient checkpointing SR-CorrNet training stores ~4.2 GiB per audio-second, so even one 4 s clip
# (~17 GiB) overflows a 16 GB T4/P100. --grad_ckpt recomputes blocks in backward: same gradients, ~0.6 GiB stored.
FT_BATCH = "--batch_size 1 --accum 4 --segment 4 --grad_ckpt"

if QUICK:
    FT_HOURS = FROZEN_HOURS = BLSTM_HOURS = 0.05
    EVAL_N = "--max_utts 40"
    STEPS = "--steps_per_epoch 40 --val_items 40"
else:
    EVAL_N = "--max_utts 500"
    STEPS = "--steps_per_epoch 150 --val_items 200"
RES = "--resume" if RESUME else ""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code(f"""
# official SR-CorrNet code, pinned to the commit this notebook was tested with (MIT licence)
!pip install -q "git+https://github.com/dmlguq456/SR_CorrNet_SS@{SRCORRNET_COMMIT}" huggingface_hub mir_eval pystoi pesq
"""),
    md("""
## Data: LibriSpeech + LibriMix metadata
About 7 GB from openslr.org, usually 5–15 minutes. Skipped for any part already present
(re-runs in the same session, or a LibriSpeech dataset attached under *Add Input*).
"""),
    code(f"""
import glob, os, subprocess
PARTS = ["dev-clean", "test-clean", "train-clean-100"]
MIRRORS = ["https://openslr.elda.org/resources/12", "https://www.openslr.org/resources/12",
           "https://us.openslr.org/resources/12"]

def present(part):
    pats = [f"{{LS_DIR}}/**/{{part}}/[0-9]*"] + [f"/kaggle/input/{{'*/' * d}}{{part}}/[0-9]*" for d in range(1, 5)]
    return any(glob.glob(p, recursive=True) for p in pats)

def fetch(part):
    for m in MIRRORS:  # stream straight into tar: no tarball kept on disk
        cmd = f"curl -fsSL --retry 5 --retry-delay 5 {{m}}/{{part}}.tar.gz | tar xz -C {{LS_DIR}}"
        if subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode == 0:
            return f"{{part}}: ok from {{m}}"
    return f"{{part}}: FAILED on all mirrors"

os.makedirs(LS_DIR, exist_ok=True)
todo = [p for p in PARTS if not present(p)]
print("already present:", [p for p in PARTS if p not in todo], "| downloading:", todo)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(3) as ex:
    for msg in ex.map(fetch, todo):
        print(msg)

# official Libri2Mix metadata (source pairs + gains), pinned commit
if not os.path.exists(f"{{META_DIR}}/metadata/Libri2Mix/libri2mix_test-clean.csv"):
    !git clone -q https://github.com/JorisCos/LibriMix {{META_DIR}} && git -C {{META_DIR}} checkout -q {LIBRIMIX_COMMIT}
!du -sh {{LS_DIR}}
"""),
    md("## Scripts (own implementation; generated from the repo's `part1_separation/` folder)"),
]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + (HERE / name).read_text(encoding="utf-8")))

cells += [
    md("## 1. Data check"),
    code("!python common.py --data_root {DATA_ROOT}"),
    md("""
## 2. Fine-tune SR-CorrNet end-to-end (runs first, so a GPU problem shows up within minutes)
Peak LR 1e-4 (5x below the from-scratch recipe) with warm-up and cosine decay sized to the budget,
4 s random crops, batch 1 x accumulation 4, gradient checkpointing, PIT SI-SDR loss + 0.5 x auxiliary decoder loss.
Epoch 0 in the log is the starting point and must match the zero-shot dev score.
"""),
    code("!python train.py --data_root {DATA_ROOT} --model srcorrnet --init pretrained --run_name ft_full --budget_h {FT_HOURS} {FT_BATCH} {STEPS} {RES}"),
    code("""
# stop here if fine-tuning crashed: otherwise best.pt is just the epoch-0 (pretrained) copy
assert os.path.exists("runs/ft_full/history.csv"), "fine-tuning did not finish an epoch, see the error above"
print(open("runs/ft_full/history.csv").read())
!python evaluate.py --data_root {DATA_ROOT} --model srcorrnet --ckpt runs/ft_full/best.pt --out results/ft_full {EVAL_N} {EVAL_EXTRA}
"""),
    md("## 3. Zero-shot baseline: the pretrained checkpoint, no training"),
    code("!python evaluate.py --data_root {DATA_ROOT} --model srcorrnet --ckpt pretrained --out results/zeroshot {EVAL_N} {EVAL_EXTRA}"),
    md("## 4. Ablation: freeze the separation encoder, adapt only split, decoder and filter heads"),
    code("""
if RUN_FROZEN:
    !python train.py --data_root {DATA_ROOT} --model srcorrnet --init pretrained --freeze_encoder --run_name ft_frozen --budget_h {FROZEN_HOURS} {FT_BATCH} {STEPS} {RES}
    !python evaluate.py --data_root {DATA_ROOT} --model srcorrnet --ckpt runs/ft_frozen/best.pt --out results/ft_frozen {EVAL_N}
"""),
    md("## 5. From-scratch baseline: own BLSTM mask network, same data, loss and evaluation"),
    code("""
if RUN_BLSTM:
    !python train.py --data_root {DATA_ROOT} --model baseline --init scratch --lr 1e-3 --batch_size 8 --accum 1 --warmup_steps 200 --run_name blstm --budget_h {BLSTM_HOURS} {STEPS} {RES}
    !python evaluate.py --data_root {DATA_ROOT} --model baseline --ckpt runs/blstm/best.pt --out results/blstm {EVAL_N}
"""),
    md("## 6. Examples, failure cases, tables and figures"),
    code("""
import os
systems = {"zeroshot": ("srcorrnet", "pretrained"), "ft_full": ("srcorrnet", "runs/ft_full/best.pt"),
           "ft_frozen": ("srcorrnet", "runs/ft_frozen/best.pt"), "blstm": ("baseline", "runs/blstm/best.pt")}
systems = {k: v for k, v in systems.items() if os.path.exists(f"results/{k}/per_utt.csv")}
RUNS = " ".join(f"{k}=results/{k}" for k in systems)
HISTS = " ".join(f"{k}=runs/{k}/history.csv" for k in systems if os.path.exists(f"runs/{k}/history.csv"))
!python analyze.py select --runs {RUNS} --main ft_full --ref zeroshot --out results/report
for k, (m, ck) in systems.items():
    !python evaluate.py --data_root {DATA_ROOT} --model {m} --ckpt {ck} --out results/{k} --ids_file results/report/example_ids.txt --save_audio
!python analyze.py report --runs {RUNS} --histories {HISTS} --main ft_full --ref zeroshot --out results/report
"""),
    code("""
from IPython.display import Markdown, Audio, Image, display
from pathlib import Path
import pandas as pd
for line in open("results/report/results.md", encoding="utf-8").read().split("\\n\\n"):
    if "![](" in line:  # render figures as images, text and tables as markdown
        display(Markdown(line.split("![](")[0])); display(Image("results/report/" + line.split("![](")[1].rstrip(")")))
    else:
        display(Markdown(line))
ex = pd.read_csv("results/report/example_ids.csv")
for _, e in ex.head(4).iterrows():
    stem = Path(e.id).stem
    display(Markdown(f"**{e.reason}**: `{e.id}`"))
    for f in ["mix", "s1_ref", "s1_est", "s2_ref", "s2_est"]:
        print(f); display(Audio(f"results/ft_full/audio/{stem}/{f}.wav"))
"""),
    code("""
# everything you need for the report, without the large checkpoints
!zip -qr part1_results.zip results runs/*/history.csv runs/*/args.json && ls -lh part1_results.zip
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
out = HERE / "kaggle_part1.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
