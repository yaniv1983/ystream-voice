"""Download the Hebrew (or another language) part of Google FLEURS as a prepared dataset.

FLEURS is read speech by native speakers with transcripts, licensed CC BY 4.0
(https://huggingface.co/datasets/google/fleurs). It is used here to teach the
model Hebrew pronunciation and rhythm; the audio is 16 kHz, which is enough
because training only uses speech tokens.

    python scripts/fetch_fleurs.py --lang he --out data/fleurs_he

Writes <out>/clips/*.wav and <out>/metadata.csv in the format of prepare_dataset.py.
"""
import argparse
import csv
import tarfile
from pathlib import Path

import soundfile
from huggingface_hub import hf_hub_download

CODES = {"he": "he_il", "en": "en_us"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="he", choices=list(CODES))
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    args = parser.parse_args()

    code = CODES[args.lang]
    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for split in args.splits:
        tsv = hf_hub_download("google/fleurs", f"data/{code}/{split}.tsv", repo_type="dataset")
        tar = hf_hub_download("google/fleurs", f"data/{code}/audio/{split}.tar.gz", repo_type="dataset")
        texts = {}
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 3:
                    texts[cols[1]] = cols[2]          # file name -> transcription with punctuation
        with tarfile.open(tar) as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in texts:
                    continue
                target = clips / f"{split}_{name}"
                if not target.exists():
                    target.write_bytes(archive.extractfile(member).read())
                info = soundfile.info(str(target))
                if not args.min_seconds <= info.duration <= args.max_seconds:
                    target.unlink()
                    skipped += 1
                    continue
                rows.append({"clip": f"clips/{target.name}", "language": args.lang, "source": f"fleurs/{split}",
                             "start": "0.00", "end": f"{info.duration:.2f}", "sample_rate": info.samplerate,
                             "text": texts[name], "flag": ""})
        print(f"{split}: {len(rows)} clips so far", flush=True)

    with (out / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hours = sum(float(r["end"]) for r in rows) / 3600
    print(f"{len(rows)} clips, {hours:.1f} hours ({skipped} outside {args.min_seconds}-{args.max_seconds} s skipped)")
    print(f"Dataset: {out}")


if __name__ == "__main__":
    main()
