"""Run scripts/try_voice.py end to end with the model replaced by a stand-in.

The script cannot be imported on a machine without torch, so torch and the
Chatterbox model are stubbed. This catches mistakes in the script itself
(undefined names, wrong arguments) before they reach a GPU machine.
"""
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
SCRIPT = Path(__file__).parent.parent / "scripts" / "try_voice.py"


class StandInModel:
    sr = 24000

    @classmethod
    def from_pretrained(cls, device, t3_model=None):
        return cls()

    def prepare_conditionals(self, path):
        pass

    def generate(self, text, language_id=None, **kwargs):
        t = np.arange(self.sr) / self.sr
        return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)[None]


@pytest.mark.parametrize("extra", [[], ["--raw"], ["--device", "cpu", "--verbose"]])
def test_try_voice_runs(tmp_path, monkeypatch, capsys, extra):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = StandInModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)

    out = tmp_path / "out.wav"
    monkeypatch.setattr(sys, "argv", ["try_voice.py", "--voice", "me.wav", "--out", str(out),
                                      "--text", "It's 7:30. [pause:0.5s] Stay tuned."] + extra)
    runpy.run_path(str(SCRIPT), run_name="__main__")

    printed = capsys.readouterr().out
    assert "Saved" in printed and "Time: load" in printed
    audio, sr = soundfile.read(out)
    assert sr == 24000 and len(audio) == int(2.65 * 24000)   # 0.15 s lead-in + 2 chunks + 0.5 s pause
