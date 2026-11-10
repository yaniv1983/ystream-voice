"""Render a tagged radio script with a cloned voice and save it as a WAV file.

    python scripts/try_voice.py --voice me.wav --text "[excited] Good morning! It's 7:30."
    python scripts/try_voice.py --voice me.wav --file script.txt --out link.wav

The language (Hebrew or English) is detected from the text unless --lang is given.
"""
import argparse
import time
from pathlib import Path

import torch
import torchaudio

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
    parser.add_argument("--check", action="store_true",
                        help="Listen to each take with Whisper (on the CPU) and redo bad ones; needs faster-whisper")
    parser.add_argument("--raw", action="store_true", help="Skip loudness mastering (-16 LUFS, -1 dBTP)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No NVIDIA GPU found: running on the CPU, which is very slow.")
    script = args.text if args.text else Path(args.file).read_text(encoding="utf-8")

    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.model)
    check = make_check(whisper_transcriber(device="cpu", compute_type="int8")) if args.check else None
    host = RadioHost(model, voice=args.voice, check=check)
    started = time.time()
    wav, report = host.render(script, language=args.lang)
    elapsed = time.time() - started
    if not args.raw:
        wav = master(wav, host.sr)

    torchaudio.save(args.out, torch.from_numpy(wav).unsqueeze(0), host.sr)
    print(f"Saved {args.out}: {report.duration:.1f} s of audio in {elapsed:.1f} s "
          f"({report.chunks} chunks, {host.sr} Hz)")
    if args.check:
        print(f"Takes rejected by the check: {report.failed_checks} (redone: {report.retries})")
    if device == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if report.fit_ratio:
        print(f"Length is {report.fit_ratio:.0%} of the [fit] target")


if __name__ == "__main__":
    main()
