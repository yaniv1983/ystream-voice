"""Compare settings for Hebrew by how intelligible the result is.

Each setting renders the same sentences several times; a speech recogniser
scores every take against the text (0 = every word understood). Prints a table
and writes all takes to <out>/<setting>/ so a person can listen to the best ones.

    python scripts/hebrew_ab.py --voice voices/me.wav --out out/hebrew_ab
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from ystream_voice import hebrew, normalize_text
from ystream_voice.verify import make_check, whisper_transcriber

# name: (model weights, niqqud on, generate() options)
SETTINGS = {
    "v3_default":   ("v3", True,  {}),
    "v3_no_niqqud": ("v3", False, {}),
    "v3_cooler":    ("v3", True,  {"temperature": 0.6}),
    "v3_strict":    ("v3", True,  {"temperature": 0.6, "cfg_weight": 0.7}),
    "v3_loose":     ("v3", True,  {"cfg_weight": 0.3}),
    "v2_default":   ("v2", True,  {}),
    "v2_no_niqqud": ("v2", False, {}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--file", default="tests/radio_scripts/he_sentences.txt")
    parser.add_argument("--out", default="out/hebrew_ab")
    parser.add_argument("--takes", type=int, default=3)
    parser.add_argument("--settings", nargs="*", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                        help="Also test a trained add-on, e.g. yaniv=adapters/yaniv (v3 weights, default options)")
    parser.add_argument("--lang", default="he", choices=["he", "en"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    sentences = [normalize_text(line.strip(), args.lang)
                 for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    check = make_check(whisper_transcriber(device="cpu", compute_type="int8"))
    out = Path(args.out)
    rows, loaded, loaded_adapter, model = [], None, None, None

    settings = {name: SETTINGS[name] + (None,) for name in args.settings}
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        settings[name] = ("v3", True, {}, path)
    names = list(settings)

    for name in names:
        weights, niqqud, options, adapter = settings[name]
        if weights != loaded or adapter or loaded_adapter:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=weights)
            if adapter:
                from ystream_voice.training import load_adapter
                load_adapter(model.t3, adapter)
            model.prepare_conditionals(args.voice)
            loaded, loaded_adapter = weights, adapter
        hebrew.set_niqqud(niqqud)
        (out / name).mkdir(parents=True, exist_ok=True)
        started = time.time()
        for s, sentence in enumerate(sentences):
            for take in range(args.takes):
                wav = model.generate(sentence, language_id=args.lang, **options)
                wav = wav.detach().cpu().numpy().reshape(-1).astype(np.float32)
                error = float(check(wav, model.sr, sentence, args.lang))
                path = out / name / f"s{s + 1:02d}_take{take + 1}.wav"
                soundfile.write(str(path), wav, model.sr, subtype="PCM_16")
                rows.append({"setting": name, "sentence": s + 1, "take": take + 1, "error": f"{error:.3f}",
                             "seconds": f"{len(wav) / model.sr:.2f}", "file": str(path), "text": sentence})
        print(f"{name}: done in {time.time() - started:.0f} s", flush=True)
    hebrew.set_niqqud(True)

    with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'setting':14} {'mean error':>10} {'perfect':>8} {'failed':>7} {'best-of-takes':>14}")
    for name in names:
        errors = [float(r["error"]) for r in rows if r["setting"] == name]
        best = [min(float(r["error"]) for r in rows if r["setting"] == name and r["sentence"] == s + 1)
                for s in range(len(sentences))]
        print(f"{name:14} {statistics.mean(errors):10.3f} {sum(e == 0 for e in errors):5d}/{len(errors):<2d} "
              f"{sum(e > 0.25 for e in errors):4d}/{len(errors):<2d} {statistics.mean(best):14.3f}")
    print("\nerror: 0 = every word understood. failed = error above 0.25.")
    print("best-of-takes = mean error if the best take of each sentence is kept.")
    print(f"All takes and results.csv are in {out}")


if __name__ == "__main__":
    main()
