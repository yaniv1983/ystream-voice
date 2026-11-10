"""Measure generation speed: load the model once, render the same script several times.

    python scripts/bench.py --voice voices/me.wav --file tests/radio_scripts/he_basic.txt --runs 5

Prints one line per run with audio length, time and GPU memory in use on the
whole card (all programs), which shows whether other apps are slowing it down.
"""
import argparse
import subprocess
import time
from pathlib import Path

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from ystream_voice import RadioHost


def card_memory_mib() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in out.split(","))
        return f"{used}/{total} MiB"
    except Exception:
        return "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model", default="v3")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    script = Path(args.file).read_text(encoding="utf-8")
    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.model)
    host = RadioHost(model, voice=args.voice)

    print(f"device {device}, torch {torch.__version__}, card memory before: {card_memory_mib()}")
    speeds = []
    for run in range(1, args.runs + 1):
        before = card_memory_mib()
        started = time.perf_counter()
        _, report = host.render(script)
        elapsed = time.perf_counter() - started
        speeds.append(elapsed / report.duration)
        print(f"run {run}: {report.duration:5.1f} s audio in {elapsed:6.1f} s  "
              f"(x{elapsed / report.duration:.2f} real time)  card memory before {before}, after {card_memory_mib()}")
    if device == "cuda":
        print(f"peak memory used by this program: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print(f"best x{min(speeds):.2f}, worst x{max(speeds):.2f} real time (below 1.00 is faster than real time)")


if __name__ == "__main__":
    main()
