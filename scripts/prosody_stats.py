"""Measure how lively a reading is: pitch movement, pauses and loudness movement.

    python scripts/prosody_stats.py real=data/yaniv/clips base=out/he_base.wav trained=out/he_joint.wav

Each NAME=PATH is a WAV file or a folder of WAVs. A flat, machine-like reading
shows a smaller pitch spread and loudness spread than the real recordings.
These numbers describe delivery only; they cannot hear accent.
"""
import argparse
import random
import sys
from pathlib import Path

import librosa
import numpy as np


def prosody(wav: np.ndarray, sr: int) -> dict:
    """Pitch spread (semitones), pitch range, share of pauses and loudness spread (dB) of one clip."""
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    f0 = f0[voiced & np.isfinite(f0)]
    rms = librosa.feature.rms(y=wav, frame_length=400, hop_length=160)[0]
    level = 20 * np.log10(np.maximum(rms, 1e-6))
    speech = level > level.max() - 35
    out = {"seconds": len(wav) / sr, "pause_share": float(1 - speech.mean()) if len(speech) else 0.0,
           "loudness_spread_db": float(level[speech].std()) if speech.any() else 0.0}
    if len(f0) >= 10:
        semitones = 12 * np.log2(f0 / np.median(f0))
        low, high = np.percentile(semitones, [5, 95])
        out.update(pitch_hz=float(np.median(f0)), pitch_spread_st=float(semitones.std()),
                   pitch_range_st=float(high - low))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", metavar="NAME=PATH")
    parser.add_argument("--max-files", type=int, default=40, help="Sample at most this many files per folder")
    args = parser.parse_args()
    keys = ["pitch_hz", "pitch_spread_st", "pitch_range_st", "pause_share", "loudness_spread_db"]
    print(f"{'group':<24}{'files':>6}{'pitch Hz':>10}{'spread st':>11}{'range st':>10}{'pauses':>8}{'loud dB':>9}")
    for spec in args.groups:
        name, _, path = spec.partition("=")
        path = Path(path)
        files = sorted(path.glob("*.wav")) if path.is_dir() else [path]
        if len(files) > args.max_files:
            files = random.Random(0).sample(files, args.max_files)
        rows = []
        for f in files:
            wav, sr = librosa.load(str(f), sr=None, mono=True)
            if len(wav) / sr >= 1.0:
                rows.append(prosody(wav, sr))
        if not rows:
            print(f"{name:<24} no usable files", file=sys.stderr)
            continue
        mean = [np.mean([r[k] for r in rows if k in r]) for k in keys]
        print(f"{name:<24}{len(rows):>6}{mean[0]:>10.0f}{mean[1]:>11.2f}{mean[2]:>10.2f}{mean[3]:>8.2f}{mean[4]:>9.2f}")


if __name__ == "__main__":
    main()
