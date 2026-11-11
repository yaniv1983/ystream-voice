"""Download the Hebrew manual-caption part of YODAS as a prepared dataset.

YODAS (espnet/yodas, licence CC BY 3.0, not gated) is YouTube speech with
captions uploaded by the video owners; the Hebrew subset is shard `iw000`.
Captions are only as good as the uploader made them, so run
filter_dataset.py afterwards to flag clips whose audio does not say the caption.

    python scripts/fetch_yodas.py --out data/web_he --max-hours 40 --max-gb 25

Writes <out>/clips/*.wav (16 kHz mono) and <out>/metadata.csv in the format of
prepare_dataset.py. `source` is the video id only.
"""
import argparse
import csv
import io
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
import soundfile
from huggingface_hub import HfApi, hf_hub_download

REPO = "espnet/yodas"
SUBSET = "iw000"                 # Hebrew (old ISO code), manual captions
LATIN = re.compile(r"[A-Za-z]")
BRACKETS = re.compile(r"[\[\](){}<>♪]")
HEBREW = re.compile(r"[֐-׿]")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[\s,.;:!?\-–]+", "", text)     # captions often start with a stray punctuation mark


def usable(text: str) -> bool:
    return bool(HEBREW.search(text)) and not LATIN.search(text) and not BRACKETS.search(text)


def parse_id(utt_id: str):
    video, _, start, end = utt_id.rsplit("-", 3)[0], *utt_id.rsplit("-", 3)[1:]
    return video, int(start) / 100, int(end) / 100


def merge_neighbours(items, min_s, max_s, gap_s):
    """items: sorted list of (start, end, text, wav) of one video. Join short neighbours separated by a small gap."""
    merged = []
    for it in items:
        if merged:
            s, e, t, w = merged[-1]
            gap = it[0] - e
            if 0 <= gap < gap_s and (e - s < min_s or it[1] - it[0] < min_s) and it[1] - s <= max_s:
                pad = np.zeros(int(round(gap * 16000)), dtype=w.dtype)
                merged[-1] = (s, it[1], f"{t} {it[2]}", np.concatenate([w, pad, it[3]]))
                continue
        merged.append(it)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-hours", type=float, default=40)
    parser.add_argument("--max-gb", type=float, default=25, help="Stop downloading after this many GB")
    parser.add_argument("--keep-free-gb", type=float, default=30)
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    args = parser.parse_args()

    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    shards = sorted((s for s in info.siblings if s.rfilename.startswith(f"data/{SUBSET}/audio/")), key=lambda s: s.rfilename)
    rows, hours, downloaded = [], 0.0, 0.0
    dropped = {"short": 0, "long": 0, "text": 0}
    for shard in shards:
        n = Path(shard.rfilename).name.split(".")[0]
        gb = (shard.size or 0) / 1e9
        free = shutil.disk_usage(out.resolve().anchor).free / 1e9
        if downloaded + gb > args.max_gb or free - gb < args.keep_free_gb or hours >= args.max_hours:
            print(f"Stopping before shard {n} (downloaded {downloaded:.1f} GB, free {free:.0f} GB, {hours:.1f} h)")
            break
        text_path = hf_hub_download(REPO, f"data/{SUBSET}/text/{n}.txt", repo_type="dataset")
        tar_path = hf_hub_download(REPO, shard.rfilename, repo_type="dataset")
        downloaded += gb
        texts = {}
        for line in open(text_path, encoding="utf-8"):
            utt, _, text = line.rstrip("\n").partition(" ")
            if utt:
                texts[utt] = clean(text)
        by_video: dict[str, list] = {}
        with tarfile.open(tar_path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                utt = Path(member.name).stem
                text = texts.get(utt, "")
                if not usable(text):
                    dropped["text"] += 1
                    continue
                wav, sr = soundfile.read(io.BytesIO(archive.extractfile(member).read()))
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    continue
                video, start, end = parse_id(utt)
                by_video.setdefault(video, []).append((start, end, text, wav.astype(np.float32)))
        for video, items in by_video.items():
            items.sort(key=lambda x: x[0])
            for i, (start, end, text, wav) in enumerate(merge_neighbours(items, args.min_seconds, args.max_seconds, 0.3)):
                dur = len(wav) / 16000
                if dur < args.min_seconds:
                    dropped["short"] += 1
                    continue
                if dur > args.max_seconds:
                    dropped["long"] += 1
                    continue
                name = f"{video}_{int(start*100):08d}.wav"
                soundfile.write(str(clips / name), wav, 16000, subtype="PCM_16")
                rows.append({"clip": f"clips/{name}", "language": "he", "source": video, "start": f"{start:.2f}",
                             "end": f"{end:.2f}", "sample_rate": 16000, "text": text, "flag": ""})
                hours += dur / 3600
        print(f"shard {n}: {len(rows)} clips, {hours:.1f} h so far, {downloaded:.1f} GB downloaded", flush=True)

    with (out / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "language", "source", "start", "end", "sample_rate", "text", "flag"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips, {hours:.1f} hours kept; dropped {dropped}; downloaded {downloaded:.1f} GB")
    print(f"Dataset: {out}")


if __name__ == "__main__":
    main()
