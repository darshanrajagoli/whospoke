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
from whospoke.corpora import (  # noqa: E402
    fetch_indicvoices, indicvoices_split_path, read_indicvoices, read_vaani_transcribed,
)
from whospoke.hinglish import romanise_word  # noqa: E402
from whospoke.languages import resolve_language  # noqa: E402
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


def sample_sets(n: int, language: str = "hi", seed: int = 0) -> dict[str, pd.DataFrame]:
    """Random utterances from each set; audio is read only for the sampled rows (keeps memory low)."""
    rng = np.random.default_rng(seed)
    lang_info = resolve_language(language)
    sets: dict[str, pd.DataFrame] = {}

    if lang_info.code == "hi":
        vfiles = sorted(Path(f) for f in glob.glob(str(RAW / "vaani_transcribed/audio/Hindi/test-*.parquet")))
        if vfiles:
            frames = []
            for i, f in enumerate(vfiles):
                d = read_vaani_transcribed(f)
                frames.append(d.assign(_file=i, _row=np.arange(len(d))))
            vaani = pd.concat(frames, ignore_index=True)
            vaani = vaani[vaani.text.str.len() > 0].reset_index(drop=True)
            sample_sets.vaani_all = vaani
            vaani = vaani.iloc[rng.choice(len(vaani), min(n, len(vaani)), replace=False)].reset_index(drop=True)
            vaani["audio_bytes"] = _take_audio(vfiles, "audio", list(zip(vaani._file, vaani._row)))
            sets["vaani-test"] = vaani
        else:
            sample_sets.vaani_all = pd.DataFrame()
    else:
        sample_sets.vaani_all = pd.DataFrame()

    iv_path = indicvoices_split_path(lang_info.code, "valid")
    if not iv_path.exists():
        try:
            iv_path = fetch_indicvoices(lang_info.code, "valid")
        except Exception:
            pass
    if iv_path.exists():
        iv = read_indicvoices(iv_path).assign(_row=lambda d: range(len(d)))
        iv = iv[(iv.text.str.len() > 0) & (iv.duration.between(1, 20))].reset_index(drop=True)
        iv = iv.iloc[rng.choice(len(iv), min(n, len(iv)), replace=False)].reset_index(drop=True)
        iv["audio_bytes"] = _take_audio([iv_path], "audio_filepath", [(0, r) for r in iv._row])
        iv["english"] = [{} for _ in range(len(iv))]
        sets[f"indicvoices-valid-{lang_info.code}"] = iv
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", "-l", default="hi", help="Language code or name (e.g. hi, bn, ta, te)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--models", nargs="+", default=["indicconformer", "indicwav2vec"])
    args = ap.parse_args()

    lang_info = resolve_language(args.language)
    sets = sample_sets(args.n, language=lang_info.code)
    if not sets:
        print(f"No audio files found for language {lang_info.name} ({lang_info.code}).")
        return
    audio = {k: [load(b) for b in df.audio_bytes] for k, df in sets.items()}
    first_set = next(iter(audio.keys()))
    rows, per_utt = [], []
    for model_name in args.models:
        try:
            asr = make_asr(model_name, language=lang_info.code)
        except Exception as exc:
            print(f"Skipping model {model_name} for language {lang_info.name}: {exc}")
            continue
        asr.transcribe([audio[first_set][0]])          # warm-up
        for set_name, df in sets.items():
            wavs = audio[set_name]
            from whospoke.pipeline import Result, _Stage

            probe = Result("", None, [], language=lang_info.code)
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
                per_utt.append({"model": model_name, "language": lang_info.code, "set": set_name, "ref": ref, "hyp": hyp})
            rows.append({
                "model": model_name, "language": lang_info.code, "set": set_name, "n": len(refs),
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

    out = PROJECT / "results"
    df = pd.DataFrame(rows)
    if not df.empty:
        suffix = f"_{lang_info.code}" if lang_info.code != "hi" else ""
        df.to_csv(out / f"asr_comparison{suffix}.csv", index=False)
        pd.DataFrame(per_utt).to_csv(out / f"asr_comparison_utterances{suffix}.csv", index=False)
        selection_set = "vaani-test" if "vaani-test" in df.set.values else df.set.iloc[0]
        best = df[df.set == selection_set].sort_values("wer").iloc[0].model
        summary = {"language": lang_info.code, "chosen_model": best, "selection_set": selection_set, "table": rows}
        (out / f"asr_comparison{suffix}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(df.round(4).to_string())
        print(f"chosen for {lang_info.name}: {best}")



if __name__ == "__main__":
    main()
