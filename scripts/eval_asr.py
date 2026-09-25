"""Stage 3 benchmark — which Hindi ASR model should the pipeline use?

Scored on two real-speech sets (single speaker, no simulation):

* **Vaani test** (IISc/ARTPARK): phone recordings from across India, spontaneous, noisy, with
  English words explicitly marked by the annotators. Neither ASR model was trained on it, so this is
  the fair set for *choosing* a model.
* **IndicVoices valid**: the speech our test conversations are built from. Reported for reference
  only — IndicConformer was trained on IndicVoices (other utterances, same speakers), which gives it
  a home advantage there.

Also measured on Vaani: how often English words spoken inside Hindi are recognised exactly
("English-word recall"), and how often our romaniser spells them the English way.

    python scripts/eval_asr.py --n 400
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.audio import SR, load  # noqa: E402
from whospoke.corpora import read_indicvoices, read_vaani_transcribed  # noqa: E402
from whospoke.hinglish import romanise_word  # noqa: E402
from whospoke.metrics import corpus_rate, matched_tokens, normalise  # noqa: E402
from whospoke.paths import RAW  # noqa: E402
from whospoke.pipeline import make_asr  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def english_norm(s: str) -> str:
    s = s.lower().replace("colour", "color").replace("litre", "liter").replace("metre", "meter")
    return re.sub(r"[^a-z]", "", s)


def _take_audio(parquet_files: list[Path], column: str, picks: list[tuple[int, int]]) -> list[bytes]:
    """Read only the chosen rows' audio bytes (file index, row index), one row group at a time."""
    import pyarrow.parquet as pq

    out: dict[tuple[int, int], bytes] = {}
    for fi, f in enumerate(parquet_files):
        rows = sorted(r for fj, r in picks if fj == fi)
        if not rows:
            continue
        pf = pq.ParquetFile(f)
        start = 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            want = [r for r in rows if start <= r < start + n]
            if want:
                col = pf.read_row_group(g, columns=[column]).column(column)
                for r in want:
                    out[(fi, r)] = col[r - start].as_py()["bytes"]
            start += n
    return [out[p] for p in picks]


def sample_sets(n: int, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Random utterances from each set; audio is read only for the sampled rows (keeps memory low)."""
    rng = np.random.default_rng(seed)
    vfiles = sorted(Path(f) for f in glob.glob(str(RAW / "vaani_transcribed/audio/Hindi/test-*.parquet")))
    frames = []
    for i, f in enumerate(vfiles):
        d = read_vaani_transcribed(f)
        frames.append(d.assign(_file=i, _row=np.arange(len(d))))
    vaani = pd.concat(frames, ignore_index=True)
    vaani = vaani[vaani.text.str.len() > 0].reset_index(drop=True)
    sample_sets.vaani_all = vaani
    vaani = vaani.iloc[rng.choice(len(vaani), min(n, len(vaani)), replace=False)].reset_index(drop=True)
    vaani["audio_bytes"] = _take_audio(vfiles, "audio", list(zip(vaani._file, vaani._row)))

    ivf = RAW / "indicvoices/hindi/valid-00000-of-00001.parquet"
    iv = read_indicvoices(ivf).assign(_row=lambda d: range(len(d)))
    iv = iv[(iv.text.str.len() > 0) & (iv.duration.between(1, 20))].reset_index(drop=True)
    iv = iv.iloc[rng.choice(len(iv), min(n, len(iv)), replace=False)].reset_index(drop=True)
    iv["audio_bytes"] = _take_audio([ivf], "audio_filepath", [(0, r) for r in iv._row])
    iv["english"] = [{} for _ in range(len(iv))]
    return {"vaani-test": vaani, "indicvoices-valid": iv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--models", nargs="+", default=["indicconformer", "indicwav2vec"])
    args = ap.parse_args()

    sets = sample_sets(args.n)
    audio = {k: [load(b) for b in df.audio_bytes] for k, df in sets.items()}
    rows, per_utt = [], []
    for model_name in args.models:
        asr = make_asr(model_name)
        asr.transcribe([audio["vaani-test"][0]])          # warm-up
        for set_name, df in sets.items():
            wavs = audio[set_name]
            from whospoke.pipeline import Result, _Stage

            probe = Result("", None, [])
            t = time.perf_counter()
            hyps = []
            with _Stage(probe, "asr"):       # device-wide GPU peak (ONNX Runtime memory included)
                for i in range(0, len(wavs), 8):
                    hyps += asr.transcribe(wavs[i: i + 8])
            elapsed = time.perf_counter() - t
            gpu_peak_mb = probe.gpu_peak_mb.get("asr", np.nan)
            refs = list(df.text)
            en_total = en_hit = 0
            for ref, hyp, eng in zip(refs, hyps, df.english):
                r_tok, h_tok = normalise(ref).split(), normalise(hyp).split()
                ok = matched_tokens(r_tok, h_tok)
                for i in eng:
                    if i < len(ok):
                        en_total += 1
                        en_hit += ok[i]
                per_utt.append({"model": model_name, "set": set_name, "ref": ref, "hyp": hyp})
            rows.append({
                "model": model_name, "set": set_name, "n": len(refs),
                "hours": round(sum(len(w) for w in wavs) / SR / 3600, 3),
                "wer": corpus_rate(list(zip(refs, hyps)), "word"),
                "cer": corpus_rate(list(zip(refs, hyps)), "char"),
                "english_word_recall": en_hit / en_total if en_total else np.nan,
                "english_words": en_total,
                "rtf": elapsed / (sum(len(w) for w in wavs) / SR),
                "gpu_device_peak_mb": gpu_peak_mb,
            })
            print(rows[-1], flush=True)
        del asr
        torch.cuda.empty_cache()

    # Romaniser, on EVERY tagged English word in the Vaani test shards (its lexicon was mined from train only):
    # does the Devanagari form come out in English spelling?
    vaani = sample_sets.vaani_all
    pairs = [(t.split()[i], e) for t, en in zip(vaani.text, vaani.english) for i, e in en.items() if i < len(t.split())]
    rom_acc = float(np.mean([english_norm(romanise_word(d)) == english_norm(e) for d, e in pairs])) if pairs else np.nan

    out = PROJECT / "results"
    df = pd.DataFrame(rows)
    df.to_csv(out / "asr_comparison.csv", index=False)
    pd.DataFrame(per_utt).to_csv(out / "asr_comparison_utterances.csv", index=False)
    best = df[df.set == "vaani-test"].sort_values("wer").iloc[0].model
    summary = {"chosen_model": best, "selection_set": "vaani-test", "romaniser_english_spelling_acc": rom_acc,
               "romaniser_pairs": len(pairs), "table": rows}
    (out / "asr_comparison.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(df.round(4).to_string())
    print(f"chosen: {best} | romaniser English-spelling accuracy on Vaani test English words: {rom_acc:.1%} ({len(pairs)} words)")


if __name__ == "__main__":
    main()
