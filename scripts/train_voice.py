"""Train a voice / language add-on (LoRA adapter) from a prepared dataset.

    python scripts/train_voice.py --data data/yaniv --out adapters/yaniv --steps 1500

`--data` is a folder made by scripts/prepare_dataset.py (clips + metadata.csv).
Rows with a flag are skipped. Clips are converted to model inputs once and
cached in <data>/features.pt. The result is adapters/yaniv.safetensors (+ .json),
usable with:  scripts/try_voice.py --adapter adapters/yaniv ...
"""
import argparse
import csv
import random
import time
from pathlib import Path

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from ystream_voice.training import (LoRAConfig, ReferencePool, TrainConfig, assign_speakers, is_validation,
                                    load_adapter, load_examples, make_example, save_examples, train)


def build_features(model, data: Path, languages) -> list:
    with (data / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if not r.get("flag") and r["text"].strip() and r["language"] in languages]
    examples = []
    for i, row in enumerate(rows, 1):
        examples.append(make_example(model, data / row["clip"], row["text"], row["language"], row["clip"]))
        if i % 20 == 0 or i == len(rows):
            print(f"  features {i}/{len(rows)}", flush=True)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, nargs="+", help="One or more prepared dataset folders")
    parser.add_argument("--out", required=True, help="Adapter path without extension")
    parser.add_argument("--languages", nargs="+", default=["he", "en"])
    parser.add_argument("--model", default="v3")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=1400, help="Lower this if the GPU runs out of memory")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-percent", type=int, default=8, help="Share of each dataset held out (stable split)")
    parser.add_argument("--max-clips", type=int, default=0, help="Use at most this many clips per dataset (0 = all)")
    parser.add_argument("--repeat", action="append", default=[], metavar="FOLDER=N",
                        help="Show one dataset's clips N times per pass, e.g. yaniv=5 (matches the folder name)")
    parser.add_argument("--multi-speaker", action="append", default=[], metavar="FOLDER",
                        help="This dataset has many voices: group its clips by voice automatically")
    parser.add_argument("--start-from", help="Adapter to merge into the model before training a new one on top")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.model)

    repeats = dict(spec.split("=") for spec in args.repeat)
    rng = random.Random(args.seed)
    train_set, val_set = [], []
    for folder in map(Path, args.data):
        cache = folder / f"features_{args.model}.pt"
        if cache.exists() and not args.rebuild_features:
            loaded = load_examples(cache)
            print(f"{folder}: {len(loaded)} cached clips")
        else:
            print(f"{folder}: converting clips to model inputs (one time)")
            loaded = build_features(model, folder, ["he", "en"])
            save_examples(loaded, cache)
        # Clips beyond the model's limits cannot be trained on.
        loaded = [e for e in loaded if e.language in args.languages
                  and 20 <= e.speech_tokens.numel() <= 1000 and e.text_tokens.numel() <= 600]
        for e in loaded:
            e.dataset = folder.name
        if folder.name in args.multi_speaker:
            voices = assign_speakers(loaded, prefix="v")
            pool = ReferencePool(loaded)
            alone = sum(not pool.has_partner(e) for e in loaded)
            loaded = [e for e in loaded if pool.has_partner(e)]
            print(f"{folder}: {voices} voices found; dropped {alone} clips whose voice appears only once")
        else:
            for e in loaded:
                e.speaker = e.language   # one person; keep references within the same language
        rng.shuffle(loaded)
        if args.max_clips:
            loaded = loaded[: args.max_clips]
        held = [e for e in loaded if is_validation(e, args.val_percent)] if len(loaded) >= 20 else []
        held_ids = {id(e) for e in held[:60]}
        kept = [e for e in loaded if id(e) not in held_ids]
        held = held[:60]
        times = int(repeats.get(folder.name, 1))
        seconds = sum(e.speech_tokens.numel() - 2 for e in kept) / 25
        print(f"{folder}: training on {len(kept)} clips ({seconds / 60:.1f} min) x{times}, holding out {len(held)}")
        train_set += kept * times
        val_set += held
    print(f"total: {len(train_set)} training clips per pass, {len(val_set)} held out; device {device}")
    if args.start_from:
        meta = load_adapter(model.t3, args.start_from)
        print(f"started from {args.start_from} (merged; best val {meta['info'].get('best', {}).get('val')})")

    # Free what training does not need.
    del model.s3gen, model.ve
    if device == "cuda":
        torch.cuda.empty_cache()

    config = TrainConfig(steps=args.steps, lr=args.lr, batch_tokens=args.batch_tokens,
                         eval_every=args.eval_every, seed=args.seed, lora=LoRAConfig(rank=args.rank,
                                                                                    alpha=2.0 * args.rank))
    started = time.time()
    summary = train(model.t3, train_set, val_set, config, args.out, log=lambda m: print(m, flush=True))
    minutes = (time.time() - started) / 60
    print(f"done in {minutes:.1f} min. Held-out speech loss: {summary['baseline_val']:.4f} before, "
          f"{summary['best']['val']:.4f} best (step {summary['best']['step']})")
    if device == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if args.start_from:
        # Remember the adapter this one was trained on top of; load_adapter applies it first.
        import json
        meta_path = Path(args.out).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["info"]["start_from"] = str(args.start_from)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Adapter: {Path(args.out).with_suffix('.safetensors')}")


if __name__ == "__main__":
    main()
