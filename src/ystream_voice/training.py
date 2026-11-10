"""Fine-tune the T3 text-to-speech-token model with LoRA adapters.

Only small adapter matrices are trained (a few MB), on top of frozen pretrained
weights, so a voice or language add-on fits an 8 GB GPU and ships as one small
file. The speech decoder (S3Gen) and vocoder are not touched.

Sequence layout, identical to inference:

    [conditioning | <text> ... </text> | <speech> s1 s2 ... </speech>]

Each position predicts the next token. Upstream's `T3.loss()` is not used: it
does not shift targets and pads text in the middle of the sequence.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from chatterbox.models.t3.modules.cond_enc import T3Cond

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update: W x + (B A x) * alpha / rank."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + update * self.scale

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_b @ self.lora_a) * self.scale
        return self.base.weight + delta.to(self.base.weight.dtype)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    targets: tuple = DEFAULT_TARGETS


def add_lora(t3, config: LoRAConfig) -> list[nn.Parameter]:
    """Freeze T3 and wrap the chosen Linear layers of its transformer. Returns trainable params."""
    for p in t3.parameters():
        p.requires_grad_(False)
    params = []
    for parent in list(t3.tfmr.modules()):
        for name, child in list(parent.named_children()):
            if name in config.targets and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, config.rank, config.alpha, config.dropout)
                wrapped.to(child.weight.device)
                setattr(parent, name, wrapped)
                params += [wrapped.lora_a, wrapped.lora_b]
    if not params:
        raise ValueError(f"No Linear layers named {config.targets} found in the transformer")
    return params


def lora_state(t3) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in t3.state_dict().items() if ".lora_a" in k or ".lora_b" in k}


def save_adapter(t3, path, config: LoRAConfig, info: dict | None = None) -> None:
    """Write the adapter as <path>.safetensors plus <path>.json (settings and training info)."""
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state(t3), str(path.with_suffix(".safetensors")))
    meta = {"lora": {**asdict(config), "targets": list(config.targets)}, "info": info or {}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_lora(t3) -> int:
    """Fold every adapter into its base weight and restore plain Linear layers. Returns the count."""
    count = 0
    for parent in list(t3.tfmr.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight.copy_(child.merged_weight())
                setattr(parent, name, child.base)
                count += 1
    return count


def load_adapter(t3, path, merge: bool = True) -> dict:
    """Apply a saved adapter to a loaded T3. With `merge`, inference runs at the original speed."""
    from safetensors.torch import load_file
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("info", {}).get("start_from"):
        # Trained on top of another adapter: that one has to be merged in first.
        load_adapter(t3, meta["info"]["start_from"], merge=True)
    lora = meta["lora"]
    add_lora(t3, LoRAConfig(lora["rank"], lora["alpha"], 0.0, tuple(lora["targets"])))
    state = load_file(str(path.with_suffix(".safetensors")))
    missing, unexpected = t3.load_state_dict(state, strict=False)
    if unexpected or any(".lora_" in k for k in missing):
        raise ValueError(f"Adapter does not match this model (unexpected {unexpected[:3]})")
    if merge:
        merge_lora(t3)
    return meta


# ------------------------------------------------------------------ data
@dataclass
class Example:
    """One training clip, already converted to model inputs (see `make_example`)."""
    text_tokens: torch.Tensor      # (Tt,) long, with start/stop text tokens
    speech_tokens: torch.Tensor    # (Ts,) long, with start/stop speech tokens
    speaker_emb: torch.Tensor      # (256,) float
    prompt_tokens: torch.Tensor    # (P,) long, speech tokens of the clip's opening, no specials
    language: str = ""
    name: str = ""
    speaker: str = ""              # clips with the same label can serve as each other's reference
    dataset: str = ""              # which prepared folder the clip came from


def frame_text_and_speech(hp, text_ids: torch.Tensor, speech_ids: torch.Tensor):
    """Add the start/stop tokens exactly as `generate()` and `T3.inference()` expect them."""
    text = torch.cat([torch.tensor([hp.start_text_token]), text_ids.long().cpu().view(-1),
                      torch.tensor([hp.stop_text_token])])
    speech = torch.cat([torch.tensor([hp.start_speech_token]), speech_ids.long().cpu().view(-1),
                        torch.tensor([hp.stop_speech_token])])
    return text, speech


def make_example(model, wav_path, text: str, language: str, name: str = "") -> Example:
    """Turn one clip + transcript into an Example using a loaded ChatterboxMultilingualTTS."""
    import librosa
    from chatterbox.mtl_tts import punc_norm
    from chatterbox.models.s3tokenizer import S3_SR
    from .normalize import normalize_text

    wav, _ = librosa.load(str(wav_path), sr=S3_SR, mono=True)
    with torch.inference_mode():
        speech_ids, _ = model.s3gen.tokenizer.forward([wav])
        speaker = torch.from_numpy(model.ve.embeds_from_wavs([wav], sample_rate=S3_SR)).mean(dim=0)
    text_ids = model.tokenizer.text_to_tokens(punc_norm(normalize_text(text, language)), language_id=language)
    speech_ids = speech_ids.view(-1).cpu()
    text_tokens, speech_tokens = frame_text_and_speech(model.t3.hp, text_ids, speech_ids)
    return Example(text_tokens, speech_tokens, speaker.float().cpu(),
                   speech_ids[: model.t3.hp.speech_cond_prompt_len].clone(), language, name)


def save_examples(examples: list[Example], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save([asdict(e) for e in examples], str(path))


def load_examples(path) -> list[Example]:
    return [Example(**d) for d in torch.load(str(path), weights_only=True)]


# ------------------------------------------------------------------ loss
def _embed(t3, example: Example, reference: Example, exaggeration: float, drop_text: bool):
    device = t3.device
    cond = T3Cond(
        speaker_emb=reference.speaker_emb.view(1, -1).to(device),
        cond_prompt_speech_tokens=reference.prompt_tokens.view(1, -1).to(device),
        emotion_adv=torch.full((1, 1, 1), float(exaggeration), device=device),
    )
    cond_emb = t3.prepare_conditioning(cond)[0]                       # (Lc, D)
    text = example.text_tokens.view(1, -1).to(device)
    speech = example.speech_tokens.view(1, -1).to(device)
    text_emb = t3.text_emb(text)
    if drop_text:  # classifier-free guidance: the unconditional branch sees no text content
        text_emb = torch.zeros_like(text_emb)
    speech_emb = t3.speech_emb(speech)
    if t3.hp.input_pos_emb == "learned":
        text_emb = text_emb + t3.text_pos_emb(text)
        speech_emb = speech_emb + t3.speech_pos_emb(speech)
    return torch.cat([cond_emb.to(text_emb.dtype), text_emb[0], speech_emb[0]]), cond_emb.size(0)


def batch_loss(t3, pairs: list[tuple[Example, Example]], *, exaggeration: float = 0.5,
               cfg_dropout: float = 0.1, text_weight: float = 0.1, return_logits: bool = False):
    """Next-token loss for a batch of (clip, reference clip) pairs.

    The reference supplies the voice conditioning (speaker embedding and prompt
    tokens), as the reference recording does at inference time. Returns
    (total, speech_loss, text_loss); with `return_logits`, the speech logits of
    the first pair instead.
    """
    embeds, spans = [], []
    for example, reference in pairs:
        drop = random.random() < cfg_dropout
        seq, len_cond = _embed(t3, example, reference, exaggeration, drop)
        embeds.append(seq)
        spans.append((len_cond, example.text_tokens.numel(), example.speech_tokens.numel(), drop))
    longest = max(e.size(0) for e in embeds)
    # Right padding only: with causal attention, padding after a sequence cannot affect it.
    batch = torch.stack([F.pad(e, (0, 0, 0, longest - e.size(0))) for e in embeds])
    hidden = t3.tfmr(inputs_embeds=batch, output_hidden_states=True, return_dict=True,
                     use_cache=False).hidden_states[-1]

    speech_losses, text_losses = [], []
    for i, ((example, _), (len_cond, n_text, n_speech, drop)) in enumerate(zip(pairs, spans)):
        text_start, speech_start = len_cond, len_cond + n_text
        # hidden[t] predicts token t+1 within each stream; the first speech token is given.
        speech_logits = t3.speech_head(hidden[i, speech_start: speech_start + n_speech - 1])
        if return_logits:
            return speech_logits
        speech_target = example.speech_tokens[1:].to(hidden.device)
        speech_losses.append(F.cross_entropy(speech_logits.float(), speech_target))
        if not drop and text_weight:
            text_logits = t3.text_head(hidden[i, text_start: text_start + n_text - 1])
            text_losses.append(F.cross_entropy(text_logits.float(), example.text_tokens[1:].to(hidden.device)))
    speech_loss = torch.stack(speech_losses).mean()
    text_loss = torch.stack(text_losses).mean() if text_losses else speech_loss.new_zeros(())
    return speech_loss + text_weight * text_loss, speech_loss, text_loss


# ------------------------------------------------------------------ training loop
@dataclass
class TrainConfig:
    steps: int = 1500
    batch_tokens: int = 1400        # speech+text tokens per batch (memory knob)
    lr: float = 1e-4
    warmup: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    cfg_dropout: float = 0.1
    text_weight: float = 0.1
    eval_every: int = 100
    seed: int = 0
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def assign_speakers(examples: list[Example], prefix: str, threshold: float = 0.75) -> int:
    """Label clips by voice when a dataset has no speaker names. Returns the number of voices found.

    Greedy clustering on the voice-encoder embedding: a clip joins the first
    cluster whose running mean it resembles (cosine above `threshold`), else
    starts a new one. Good enough to find "another clip by the same person".
    """
    centres: list[torch.Tensor] = []
    counts: list[int] = []
    for e in examples:
        emb = F.normalize(e.speaker_emb.float(), dim=0)
        best, best_sim = -1, threshold
        for i, centre in enumerate(centres):
            sim = float(torch.dot(emb, F.normalize(centre, dim=0)))
            if sim > best_sim:
                best, best_sim = i, sim
        if best < 0:
            centres.append(emb.clone())
            counts.append(1)
            best = len(centres) - 1
        else:
            centres[best] = (centres[best] * counts[best] + emb) / (counts[best] + 1)
            counts[best] += 1
        e.speaker = f"{prefix}{best:04d}"
    return len(centres)


def is_validation(example: Example, percent: int = 8) -> bool:
    """Stable split by clip name, so the same clips are held out in every run."""
    import zlib
    return zlib.crc32(f"{example.dataset}/{example.name}".encode("utf-8")) % 100 < percent


class ReferencePool:
    """Finds, for a clip, another clip by the same speaker to act as the voice reference."""

    def __init__(self, examples: list[Example]):
        self.by_speaker: dict[tuple, list[Example]] = {}
        for e in examples:
            self.by_speaker.setdefault((e.dataset, e.speaker), []).append(e)

    def has_partner(self, example: Example) -> bool:
        return len(self.by_speaker.get((example.dataset, example.speaker), ())) > 1

    def pick(self, example: Example, rng: random.Random) -> Example:
        """Another clip by the same speaker, so the model cannot copy its own prompt."""
        group = self.by_speaker.get((example.dataset, example.speaker), ())
        others = [e for e in group if e is not example]
        return rng.choice(others) if others else example


def pick_reference(example: Example, pool, rng: random.Random) -> Example:
    pool = pool if isinstance(pool, ReferencePool) else ReferencePool(pool)
    return pool.pick(example, rng)


def make_batches(examples: list[Example], batch_tokens: int, rng: random.Random) -> list[list[Example]]:
    """Group clips of similar length; each batch holds at most `batch_tokens` after padding."""
    order = sorted(examples, key=lambda e: e.speech_tokens.numel() + e.text_tokens.numel() + rng.random())
    batches, current, longest = [], [], 0
    for e in order:
        size = e.speech_tokens.numel() + e.text_tokens.numel()
        if current and max(longest, size) * (len(current) + 1) > batch_tokens:
            batches.append(current)
            current, longest = [], 0
        current.append(e)
        longest = max(longest, size)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(t3, examples: list[Example], pool, batch_tokens: int) -> float:
    """Mean speech loss over held-out clips, with fixed references and no text dropout."""
    if not examples:
        return float("nan")
    pool = pool if isinstance(pool, ReferencePool) else ReferencePool(pool)
    rng = random.Random(1234)
    was_training = t3.training
    t3.eval()
    total, count = 0.0, 0
    amp = t3.device.type == "cuda"
    for batch in make_batches(examples, batch_tokens, random.Random(0)):
        pairs = [(e, pool.pick(e, rng)) for e in batch]
        with torch.autocast(device_type=t3.device.type, dtype=torch.bfloat16, enabled=amp):
            _, speech_loss, _ = batch_loss(t3, pairs, cfg_dropout=0.0, text_weight=0.0)
        total += float(speech_loss) * len(batch)
        count += len(batch)
    t3.train(was_training)
    return total / count


def evaluate_by_dataset(t3, examples: list[Example], pool, batch_tokens: int) -> dict[str, float]:
    """Held-out loss per dataset, plus "mean": the average of the per-dataset values."""
    names = sorted({e.dataset for e in examples})
    result = {n: evaluate(t3, [e for e in examples if e.dataset == n], pool, batch_tokens) for n in names}
    result["mean"] = sum(result.values()) / len(result) if result else float("nan")
    return result


def train(t3, train_set: list[Example], val_set: list[Example], config: TrainConfig, out_path,
          log=print, use_amp: bool | None = None) -> dict:
    """Train a LoRA adapter on `train_set`; keep the adapter with the best held-out loss.

    Returns a summary dict (also stored next to the adapter). `t3` is left with
    unmerged adapters holding the *last* weights; reload the saved file to use the best.
    """
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    params = add_lora(t3, config.lora)
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def lr_at(step):
        if step < config.warmup:
            return (step + 1) / config.warmup
        progress = (step - config.warmup) / max(1, config.steps - config.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    device = t3.device
    amp = (device.type == "cuda") if use_amp is None else use_amp
    # Held-out clips may still lend their voice as a reference; they are never trained on.
    pool = ReferencePool(train_set + val_set)

    def run_eval():
        detail = evaluate_by_dataset(t3, val_set, pool, config.batch_tokens)
        return detail["mean"], detail

    def describe(detail):
        return ", ".join(f"{k or 'data'} {v:.4f}" for k, v in detail.items() if k != "mean")

    base_val, base_detail = run_eval()
    log(f"step 0: val speech loss {base_val:.4f} [{describe(base_detail)}] (before training)")
    history = [{"step": 0, "val": base_val, "val_by_dataset": base_detail}]
    best = {"step": 0, "val": base_val, "val_by_dataset": base_detail}
    info = {"clips": len(train_set), "val_clips": len(val_set), "config": asdict(config)}

    t3.train()
    step, running = 0, []
    while step < config.steps:
        for batch in make_batches(train_set, config.batch_tokens, rng):
            if step >= config.steps:
                break
            pairs = [(e, pool.pick(e, rng)) for e in batch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss, speech_loss, _ = batch_loss(t3, pairs, cfg_dropout=config.cfg_dropout,
                                                  text_weight=config.text_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running.append(float(speech_loss.detach()))
            if step % config.eval_every == 0 or step == config.steps:
                val, detail = run_eval()
                train_loss = sum(running) / len(running)
                running = []
                history.append({"step": step, "train": train_loss, "val": val, "val_by_dataset": detail})
                log(f"step {step}: train speech loss {train_loss:.4f}, val {val:.4f} [{describe(detail)}]")
                # Without held-out clips there is nothing to select on: keep the latest.
                if math.isnan(val) or val < best["val"] or math.isnan(best["val"]):
                    best = {"step": step, "val": val, "val_by_dataset": detail}
                    save_adapter(t3, out_path, config.lora, {**info, "best": best, "history": history})
    summary = {**info, "best": best, "baseline_val": base_val, "baseline_by_dataset": base_detail,
               "history": history}
    Path(out_path).with_suffix(".history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
