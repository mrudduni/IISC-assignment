"""Separate any mono/stereo wav (or a folder of wavs) into two sources.

  python inference.py --ckpt runs/ft_full/best.pt --input mixture.wav --output separated/
  python inference.py --ckpt pretrained --input some_dir/ --output separated/

Input is resampled to 8 kHz (the model's rate); outputs are written at 8 kHz as
<name>_s1.wav and <name>_s2.wav. Output order is arbitrary (permutation-invariant model).
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from common import SR, load_model, load_wav, resample, separate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["srcorrnet", "baseline"], default="srcorrnet")
    ap.add_argument("--ckpt", default="pretrained")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.model, a.ckpt).to(device).eval()
    inp = Path(a.input)
    files = sorted(inp.glob("*.wav")) if inp.is_dir() else [inp]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for f in files:
        x, sr = load_wav(str(f))
        x = resample(x, sr, SR)
        est = separate(model, x, device)
        peak = est.abs().max().clamp(min=1e-8)
        est = est / peak * 0.9 if peak > 0.9 else est
        for i in range(2):
            sf.write(out / f"{f.stem}_s{i+1}.wav", est[i].numpy(), SR)
        print(f"{f.name} -> {f.stem}_s1.wav, {f.stem}_s2.wav")


if __name__ == "__main__":
    main()
