"""Training mechanics on a tiny randomly initialised T3 (no pretrained weights needed)."""
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chatterbox.models.t3 import T3                                    # noqa: E402
from chatterbox.models.t3 import llama_configs                         # noqa: E402
from chatterbox.models.t3.modules.cond_enc import T3Cond               # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config            # noqa: E402
from ystream_voice import training as tr                               # noqa: E402


@pytest.fixture()
def t3():
    tiny = dict(llama_configs.LLAMA_CONFIGS["Llama_520M"], hidden_size=1024, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, head_dim=256)
    llama_configs.LLAMA_CONFIGS["Llama_tiny"] = tiny
    hp = T3Config.multilingual()
    hp.llama_config_name = "Llama_tiny"
    torch.manual_seed(0)
    model = T3(hp).float().eval()
    yield model
    del llama_configs.LLAMA_CONFIGS["Llama_tiny"]


def example(hp, n_text, n_speech, seed, language="he"):
    g = torch.Generator().manual_seed(seed)
    text, speech = tr.frame_text_and_speech(hp, torch.randint(1, 250, (n_text,), generator=g),
                                            torch.randint(0, 6561, (n_speech,), generator=g))
    return tr.Example(text, speech, torch.randn(256, generator=g), speech[1:41].clone(), language, f"clip{seed}")


def test_framing_matches_inference_tokens(t3):
    e = example(t3.hp, 5, 7, 0)
    assert e.text_tokens[0] == t3.hp.start_text_token and e.text_tokens[-1] == t3.hp.stop_text_token
    assert e.speech_tokens[0] == t3.hp.start_speech_token and e.speech_tokens[-1] == t3.hp.stop_speech_token
    assert e.text_tokens.numel() == 7 and e.speech_tokens.numel() == 9


def test_logits_match_upstream_forward_layout(t3):
    """Our sequence layout must be the one the pretrained model was built for."""
    e, ref = example(t3.hp, 6, 12, 1), example(t3.hp, 4, 50, 2)
    ours = tr.batch_loss(t3, [(e, ref)], cfg_dropout=0.0, return_logits=True)
    cond = T3Cond(speaker_emb=ref.speaker_emb.view(1, -1), cond_prompt_speech_tokens=ref.prompt_tokens.view(1, -1),
                  emotion_adv=torch.full((1, 1, 1), 0.5))
    with torch.no_grad():
        out = t3.forward(t3_cond=cond, text_tokens=e.text_tokens.view(1, -1),
                         text_token_lens=torch.tensor([e.text_tokens.numel()]),
                         speech_tokens=e.speech_tokens.view(1, -1),
                         speech_token_lens=torch.tensor([e.speech_tokens.numel()]), training=True)
    assert torch.allclose(ours, out.speech_logits[0, :-1], atol=1e-4)


def test_padding_does_not_change_a_sequence(t3):
    short, long, ref = example(t3.hp, 4, 10, 3), example(t3.hp, 9, 40, 4), example(t3.hp, 4, 50, 5)
    with torch.no_grad():
        alone = tr.batch_loss(t3, [(short, ref)], cfg_dropout=0.0, text_weight=0.0)[1]
        both = tr.batch_loss(t3, [(short, ref), (long, ref)], cfg_dropout=0.0, text_weight=0.0)[1]
        long_alone = tr.batch_loss(t3, [(long, ref)], cfg_dropout=0.0, text_weight=0.0)[1]
    assert torch.allclose(both, (alone + long_alone) / 2, atol=1e-4)


def test_lora_starts_as_identity_then_merges_and_reloads(t3, tmp_path):
    e, ref = example(t3.hp, 6, 12, 1), example(t3.hp, 4, 50, 2)
    with torch.no_grad():
        before = tr.batch_loss(t3, [(e, ref)], cfg_dropout=0.0, return_logits=True)
    config = tr.LoRAConfig(rank=4, alpha=8, dropout=0.0)
    params = tr.add_lora(t3, config)
    assert len(params) == 2 * 2 * 4                      # 2 layers x 4 projections x (A, B)
    assert all(p.requires_grad for p in params)
    assert sum(p.requires_grad for p in t3.parameters()) == len(params)
    with torch.no_grad():
        assert torch.allclose(tr.batch_loss(t3, [(e, ref)], cfg_dropout=0.0, return_logits=True), before, atol=1e-5)
        for p in params:                                 # pretend training happened
            p.add_(0.05 * torch.randn_like(p))
        changed = tr.batch_loss(t3, [(e, ref)], cfg_dropout=0.0, return_logits=True)
    assert not torch.allclose(changed, before, atol=1e-3)

    tr.save_adapter(t3, tmp_path / "voice", config, {"note": "test"})
    assert tr.merge_lora(t3) == 8
    with torch.no_grad():
        merged = tr.batch_loss(t3, [(e, ref)], cfg_dropout=0.0, return_logits=True)
    assert torch.allclose(merged, changed, atol=1e-4)

    # A fresh model with the adapter loaded behaves like the trained one.
    torch.manual_seed(0)
    fresh = T3(t3.hp).float().eval()
    meta = tr.load_adapter(fresh, tmp_path / "voice")
    assert meta["info"]["note"] == "test"
    assert not any(isinstance(m, tr.LoRALinear) for m in fresh.modules())
    with torch.no_grad():
        assert torch.allclose(tr.batch_loss(fresh, [(e, ref)], cfg_dropout=0.0, return_logits=True), changed, atol=1e-4)


def test_batches_respect_token_budget(t3):
    examples = [example(t3.hp, 5 + i, 20 + 10 * i, i) for i in range(12)]
    batches = tr.make_batches(examples, 300, random.Random(0))
    assert sorted(e.name for b in batches for e in b) == sorted(e.name for e in examples)
    for b in batches:
        longest = max(e.speech_tokens.numel() + e.text_tokens.numel() for e in b)
        assert longest * len(b) <= 300 or len(b) == 1


def test_training_reduces_loss_and_saves_best_adapter(t3, tmp_path):
    examples = [example(t3.hp, 6, 30, i) for i in range(6)]
    config = tr.TrainConfig(steps=30, batch_tokens=200, lr=5e-3, warmup=2, eval_every=10, cfg_dropout=0.1,
                            lora=tr.LoRAConfig(rank=8, alpha=16, dropout=0.0))
    logs = []
    summary = tr.train(t3, examples, examples[:2], config, tmp_path / "a", log=logs.append, use_amp=False)
    assert summary["best"]["val"] < summary["baseline_val"] - 0.05
    assert (tmp_path / "a.safetensors").exists() and (tmp_path / "a.json").exists()
    assert (tmp_path / "a.history.json").exists() and len(logs) == 4
