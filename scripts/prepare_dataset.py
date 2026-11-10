"""Cut voice recordings into short clips with transcripts, ready for training.

    python scripts/prepare_dataset.py --lang he --out data/yaniv  "C:/VO/Script *.wav"
    python scripts/prepare_dataset.py --lang en --out data/yaniv  "C:/VO/yStream*.wav"

Run once per language into the same --out folder. The originals are only read.
Result: <out>/clips/*.wav (original sample rate) and <out>/metadata.csv, which
opens in Excel so the transcripts can be reviewed and corrected.
"""
import argparse
import glob
from pathlib import Path

from ystream_voice.dataset import prepare_dataset
from ystream_voice.verify import whisper_transcriber


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+", help="Audio files or wildcard patterns")
    parser.add_argument("--lang", required=True, choices=["he", "en"])
    parser.add_argument("--out", required=True, help="Dataset folder to create or add to")
    parser.add_argument("--no-transcribe", action="store_true", help="Only cut clips, leave the text empty")
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    files = sorted({Path(p) for pattern in args.patterns for p in glob.glob(pattern)})
    if not files:
        parser.error("No files matched")
    transcribe = None
    if not args.no_transcribe:
        compute = "int8" if args.whisper_device == "cpu" else "default"
        transcribe = whisper_transcriber(device=args.whisper_device, compute_type=compute)

    rows = prepare_dataset(files, Path(args.out), args.lang, transcribe)
    seconds = sum(float(r["end"]) - float(r["start"]) for r in rows)
    print(f"{len(files)} recordings -> {len(rows)} clips, {seconds / 60:.1f} minutes ({args.lang})")
    print(f"Transcripts: {Path(args.out) / 'metadata.csv'}")


if __name__ == "__main__":
    main()
