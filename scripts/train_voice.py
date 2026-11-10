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
from ystream_voice.training import (LoRAConfig, TrainConfig, load_examples, make_example,
                                    save_examples, train)


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
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-clips", type=int, default=0, help="Use only this many clips (0 = all)")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.model)

    examples = []
    for folder in map(Path, args.data):
        cache = folder / f"features_{args.model}.pt"
        if cache.exists() and not args.rebuild_features:
            loaded = [e for e in load_examples(cache) if e.language in args.languages]
            print(f"{folder}: {len(loaded)} cached clips")
        else:
            print(f"{folder}: converting clips to model inputs (one time)")
            loaded = build_features(model, folder, args.languages)
            save_examples(loaded, cache)
        examples += loaded
    # Clips beyond the model's limits cannot be trained on.
    hp = model.t3.hp
    examples = [e for e in examples if 20 <= e.speech_tokens.numel() <= 1000 and e.text_tokens.numel() <= 600]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.max_clips:
        examples = examples[: args.max_clips]
    n_val = max(2, int(len(examples) * args.val_fraction)) if len(examples) >= 20 else 0
    val_set, train_set = examples[:n_val], examples[n_val:]
    seconds = sum(e.speech_tokens.numel() - 2 for e in train_set) / 25
    print(f"training on {len(train_set)} clips ({seconds / 60:.1f} min), validating on {len(val_set)}; "
          f"device {device}")

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
    print(f"Adapter: {Path(args.out).with_suffix('.safetensors')}")


if __name__ == "__main__":
    main()
