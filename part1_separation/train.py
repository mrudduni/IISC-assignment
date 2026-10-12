"""Fine-tune SR-CorrNet (or train the BLSTM baseline) on Libri2Mix.

Own implementation of the training loop. Examples:
  python train.py --model srcorrnet --init pretrained --run_name ft_full
  python train.py --model srcorrnet --init pretrained --freeze_encoder --run_name ft_frozen_enc
  python train.py --model baseline --init scratch --lr 1e-3 --run_name blstm
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (Libri2MixDataset, build_model, find_libri2mix, mag_snr_loss, pit_si_sdr,
                    apply_perm, seed_everything, si_sdr)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/kaggle/input")
    ap.add_argument("--mix_type", default="mix_clean", help="mix_clean = two speakers, no noise")
    ap.add_argument("--model", choices=["srcorrnet", "baseline"], default="srcorrnet")
    ap.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing for SR-CorrNet (needed on 16 GB GPUs; same gradients, ~30%% slower)")
    ap.add_argument("--run_name", default="ft_full")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10, help="ignored when --budget_h is set")
    ap.add_argument("--budget_h", type=float, default=None,
                    help="wall-clock training budget; epochs and the LR schedule are sized to fit it")
    ap.add_argument("--steps_per_epoch", type=int, default=2000, help="optimizer steps per 'epoch'")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--segment", type=float, default=4.0, help="training crop length in seconds")
    ap.add_argument("--aux_weight", type=float, default=0.5, help="SR-CorrNet decoder aux loss")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--val_items", type=int, default=500)
    ap.add_argument("--val_segment", type=float, default=6.0)
    ap.add_argument("--time_budget_h", type=float, default=11.0, help="hard stop before Kaggle's 12 h limit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    vals, vals_in = [], []
    for mix, srcs in loader:
        mix, srcs = mix.to(device), srcs.to(device)
        scale = mix.std(-1, keepdim=True) + 1e-8
        est = model(mix / scale)
        best, _ = pit_si_sdr(est, srcs)
        vals.append(best.mean(-1).cpu())
        vals_in.append(si_sdr(mix[:, None].expand_as(srcs), srcs).mean(-1).cpu())
    v, vi = torch.cat(vals), torch.cat(vals_in)
    return {"val_si_sdr": v.mean().item(), "val_si_sdri": (v - vi).mean().item()}


def main():
    a = parse()
    seed_everything(a.seed)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    print("args:", json.dumps(vars(a)))

    splits = find_libri2mix(a.data_root, a.mix_type)
    assert "train" in splits and "dev" in splits, f"need train and dev splits, found {list(splits)}"
    for k, v in splits.items():
        print(f"{k}: {len(v['ids'])} mixtures @ {v['sr']} Hz from {v['dir']}")

    train_ds = Libri2MixDataset(splits["train"], a.segment, train=True)
    val_ds = Libri2MixDataset(splits["dev"], a.val_segment, train=False, max_items=a.val_items)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers,
                          drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, a.batch_size, shuffle=False, num_workers=a.workers)

    model = build_model(a.model, a.init).to(device)
    if a.grad_ckpt:
        assert a.model == "srcorrnet"
        model.enable_grad_checkpointing()
    if a.freeze_encoder:
        assert a.model == "srcorrnet"
        for p in model.encoder_params():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: {n_train/1e6:.2f}M trainable / {n_all/1e6:.2f}M total")

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    plan = {"epochs": a.epochs, "total": a.epochs * a.steps_per_epoch}  # resized if --budget_h
    def lr_lambda(step):  # linear warm-up then cosine decay to 1% of peak
        if step < a.warmup_steps:
            return (step + 1) / a.warmup_steps
        p = (step - a.warmup_steps) / max(1, plan["total"] - a.warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history, start_epoch, step, best = [], 1, 0, -1e9
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, step, best, history = ck["epoch"] + 1, ck["step"], ck["best"], ck["history"]
        if a.budget_h is not None:  # keep the budget-sized schedule; otherwise --epochs may be extended
            plan.update(ck.get("plan", {}))
        print(f"resumed from epoch {ck['epoch']} (best val SI-SDRi {best:.2f} dB)")
    else:
        # Epoch 0 = the starting point. For --init pretrained this is the zero-shot score and
        # doubles as proof that the pretrained weights loaded correctly.
        t_val = time.time()
        v = validate(model, val_dl, device)
        val_seconds = time.time() - t_val
        history.append({"epoch": 0, "step": 0, "train_loss": float("nan"), **v, "lr": 0.0, "minutes": 0.0})
        print(f"[epoch 0] val SI-SDRi {v['val_si_sdri']:.2f} dB")
        best = v["val_si_sdri"]  # fine-tuning must beat the starting point to replace best.pt
        torch.save({"model": model.state_dict()}, out / "best.pt")

    use_aux = a.model == "srcorrnet" and a.aux_weight > 0
    probe = a.budget_h is not None and start_epoch == 1
    it = iter(train_dl)
    epoch = start_epoch - 1
    while epoch < plan["epochs"]:
        epoch += 1
        model.train()
        if a.freeze_encoder:  # keep frozen blocks in eval mode (no dropout/statistics drift)
            model.net.encoder.eval(); model.net.enc_block.eval()
        tot, n = 0.0, 0
        pbar = tqdm(range(a.steps_per_epoch), desc=f"epoch {epoch}", mininterval=30)
        for i_step in pbar:
            if probe and i_step == 5:
                t_probe = time.time()
            if probe and i_step == 35:  # measure speed, then size epochs + cosine to the budget
                step_s = (time.time() - t_probe) / 30
                per_epoch = a.steps_per_epoch * step_s + val_seconds
                remaining = a.budget_h * 3600 - (t_probe - t_start) + 5 * step_s
                plan["epochs"] = max(1, int(remaining // per_epoch))
                plan["total"] = plan["epochs"] * a.steps_per_epoch
                probe = False
                print(f"\n[budget] {step_s:.2f} s/step, {per_epoch/60:.1f} min/epoch -> "
                      f"{plan['epochs']} epochs = {plan['total']} steps in {a.budget_h} h")
            opt.zero_grad(set_to_none=True)
            for _ in range(a.accum):
                try:
                    mix, srcs = next(it)
                except StopIteration:
                    it = iter(train_dl); mix, srcs = next(it)
                mix, srcs = mix.to(device, non_blocking=True), srcs.to(device, non_blocking=True)
                scale = mix.std(-1, keepdim=True) + 1e-8      # same normalisation as the official engine
                mix, srcs = mix / scale, srcs / scale[..., None]
                est, est_aux = model(mix, aux=True)
                best_sdr, swapped = pit_si_sdr(est, srcs)
                loss = -best_sdr.mean()
                if use_aux and est_aux:
                    # aux outputs use the permutation chosen for the final output, as in the repo
                    l_aux = sum(mag_snr_loss(apply_perm(e, swapped), srcs) for e in est_aux) / len(est_aux)
                    loss = loss + a.aux_weight * l_aux
                (loss / a.accum).backward()
                tot += -best_sdr.mean().item(); n += 1
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step(); sched.step(); step += 1
            pbar.set_postfix(train_si_sdr=f"{-tot/n:.2f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        v = validate(model, val_dl, device)
        minutes = (time.time() - t_start) / 60
        row = {"epoch": epoch, "step": step, "train_loss": tot / n, **v,
               "lr": sched.get_last_lr()[0], "minutes": minutes}
        history.append(row)
        mem = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
        print(f"[epoch {epoch}] train SI-SDR {-tot/n:.2f} dB | val SI-SDRi {v['val_si_sdri']:.2f} dB | {minutes:.0f} min{mem}")
        if v["val_si_sdri"] > best:
            best = v["val_si_sdri"]
            torch.save({"model": model.state_dict()}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "best": best, "history": history, "plan": plan},
                   out / "last.pt")
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)
        per_epoch_h = (time.time() - t_start) / 3600 / max(1, epoch - start_epoch + 1)
        if (time.time() - t_start) / 3600 + per_epoch_h > a.time_budget_h:
            print(f"stopping: next epoch would exceed the {a.time_budget_h} h budget")
            break

    print(f"done. best val SI-SDRi {best:.2f} dB -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
