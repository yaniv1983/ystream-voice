"""Render a tagged radio script with a cloned voice and save it as a WAV file.

    python scripts/try_voice.py --voice me.wav --text "[excited] Good morning! It's 7:30."
    python scripts/try_voice.py --voice me.wav --file script.txt --out link.wav

The language (Hebrew or English) is detected from the text unless --lang is given.
"""
import argparse
import logging
import time
from pathlib import Path

import soundfile
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from ystream_voice import RadioHost, master, make_check, whisper_transcriber


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", required=True, help="Clean recording of the voice to clone (10-30 s, no music)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Script text, tags allowed")
    source.add_argument("--file", help="UTF-8 text file with the script")
    parser.add_argument("--lang", choices=["he", "en"], help="Force a language instead of detecting it")
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--model", default="v3", help="Multilingual T3 weights: v2 or v3")
    parser.add_argument("--adapter", help="Trained voice add-on (path without extension) from train_voice.py")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--verbose", action="store_true", help="Print what the checker heard for each take")
    parser.add_argument("--check", action="store_true",
                        help="Listen to each take with Whisper (on the CPU) and redo bad ones; needs faster-whisper")
    parser.add_argument("--raw", action="store_true", help="Skip loudness mastering (-16 LUFS, -1 dBTP)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    if device == "cpu":
        print("Running on the CPU, which is slower than a GPU.")
    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("ystream_voice").setLevel(logging.INFO)
    script = args.text if args.text else Path(args.file).read_text(encoding="utf-8")

    started = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.model)
    if args.adapter:
        from ystream_voice.training import load_adapter
        load_adapter(model.t3, args.adapter)
    check = make_check(whisper_transcriber(device="cpu", compute_type="int8")) if args.check else None
    host = RadioHost(model, voice=args.voice, check=check, lead_in=0.15)
    load_seconds = time.time() - started
    started = time.time()
    wav, report = host.render(script, language=args.lang)
    elapsed = time.time() - started
    if not args.raw:
        wav = master(wav, host.sr)

    soundfile.write(args.out, wav, host.sr, subtype="PCM_24")
    print(f"Saved {args.out}: {report.duration:.1f} s of audio in {elapsed:.1f} s "
          f"({report.chunks} chunks, {host.sr} Hz, device {device})")
    print(f"Time: load {load_seconds:.1f} s, voice generation {report.generate_seconds:.1f} s, "
          f"checking {report.check_seconds:.1f} s")
    if args.check:
        print(f"Takes rejected by the check: {report.failed_checks} (redone: {report.retries}); "
              f"worst kept chunk error {report.worst_error:.2f}")
    if device == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if report.fit_ratio:
        print(f"Length is {report.fit_ratio:.0%} of the [fit] target")


if __name__ == "__main__":
    main()
