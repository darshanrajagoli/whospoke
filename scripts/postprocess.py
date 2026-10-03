#!/usr/bin/env python3
"""Run Stage 4 (LLM report) on an existing Stage-3 transcript.json. Same as ``python -m whospoke postprocess``.

    python scripts/serve_llm.py                                   # once, in another terminal
    python scripts/postprocess.py results/demo/transcript.json    # writes report.md + report.json next to it

Options: ``python scripts/postprocess.py --help`` (see docs/MILESTONE4.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.insert(1, "postprocess")
    main()
