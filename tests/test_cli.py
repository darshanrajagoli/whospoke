"""Command line: `whospoke separate` writes one .wav per separated voice; output is UTF-8 on any console."""
import subprocess
import sys

import numpy as np

from whospoke import __main__ as cli
from whospoke import separation
from whospoke.audio import SR, load, save


class _HalfSeparator:
    """Stand-in separator: track 1 = the input, track 2 = twice the input (exceeds full scale on purpose)."""

    def __init__(self, name):
        self.name = name

    def separate(self, wav):
        return np.stack([wav, 2.0 * wav])


def test_separate_writes_one_wav_per_track_without_clipping(tmp_path, monkeypatch):
    t = np.arange(2 * SR) / SR
    wav = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    save(tmp_path / "in.wav", wav)
    monkeypatch.setattr(separation, "Separator", _HalfSeparator)
    monkeypatch.setattr(sys, "argv", ["whospoke", "separate", str(tmp_path / "in.wav"), "--out", str(tmp_path / "out")])
    cli.main()
    a, b = load(tmp_path / "out" / "separated_track_1.wav"), load(tmp_path / "out" / "separated_track_2.wav")
    assert len(a) == len(b) == len(wav)
    assert np.abs(b).max() <= 1.0                       # scaled below full scale, not clipped
    assert np.allclose(b, 2 * a, atol=2e-4)             # one common factor keeps the tracks' balance


def test_cli_output_is_utf8_when_redirected():
    # a Windows console redirected to a pipe defaults to cp1252, which cannot print Devanagari
    code = ("import sys; sys.argv = ['whospoke', 'postprocess', '--help']; from whospoke.__main__ import main\n"
            "try:\n    main()\nexcept SystemExit:\n    pass\nprint('\\u0939\\u093f\\u0902\\u0926\\u0940')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, env={**__import__("os").environ,
                         "PYTHONIOENCODING": "cp1252", "PYTHONPATH": str(__import__("pathlib").Path(cli.__file__).parents[1])})
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert "हिंदी".encode("utf-8") in out.stdout
