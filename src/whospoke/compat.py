"""Small shims so the pinned third-party code runs on newer torch / torchaudio (e.g. Google Colab, late 2026).

The project was built and measured on torch/torchaudio 2.5.1 (requirements.txt), where every shim here is a no-op.

* torchaudio >= 2.9 dropped ``AudioMetaData``, ``list_audio_backends`` and friends. pyannote.audio 3.3 names them at
  import time. WhoSpoke always hands pyannote in-memory waveforms, so pyannote never reads a file through torchaudio:
  harmless stand-ins are enough.
* torch >= 2.6 loads checkpoints with ``weights_only=True`` by default, which refuses the pyannote and Asteroid
  checkpoints (they store config objects next to the weights). Those checkpoints come from the pinned Hugging Face
  repositories the project already trusts, so the old default is restored for them.
"""
from __future__ import annotations

import os


def apply() -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torchaudio
    except Exception:          # torchaudio missing entirely: nothing to patch
        return
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:   # noqa: D401 - stand-in type used only in annotations
            pass

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
