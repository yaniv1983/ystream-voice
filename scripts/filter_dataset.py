"""Flag clips whose audio does not match their text (for datasets with their own captions).

    python scripts/filter_dataset.py --data data/yodas_he --max-error 0.15

Every unflagged clip in <data>/metadata.csv is transcribed and compared with
its text; mismatches get the flag "mismatch" and are skipped by training.
Safe to stop and rerun: checked clips are remembered.
"""
import argparse

from ystream_voice.dataset import flag_mismatches
from ystream_voice.verify import whisper_transcriber


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-error", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    compute = "int8" if args.device == "cpu" else "default"
    transcribe = whisper_transcriber(device=args.device, compute_type=compute)
    checked, flagged = flag_mismatches(args.data, transcribe, args.max_error, log=lambda m: print(m, flush=True))
    print(f"checked {checked} clips, flagged {flagged} as mismatch")


if __name__ == "__main__":
    main()
