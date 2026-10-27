"""whospoke live: two people talk into one laptop microphone, and who said what builds up on screen.

    python demo/live.py                   # opens http://127.0.0.1:8765 in the browser
    python demo/live.py --list-mics       # then pick one with --mic <number>

While people talk, the recording so far goes through the pipeline again whenever ~1.5 s of new audio has arrived
(Order B: diarize, separate the overlaps, transcribe), so the timeline and the transcript fill in and settle as speech
accumulates. Who is who comes from clustering the whole recording, so during the first ~10 s the system may see one
speaker or swap labels; that is the clustering deciding, not a fault. After Stop, one final pass runs on the full
recording, the two voices are pulled apart for playback (as on the flash-talk slide), and, for Hindi, the Stage-4
LLM can write its report.

The speech models live in a separate process. Stages 1-3 (~4.5 GB) and the 4-bit LLM (5.9 GB) do not fit on the
6 GB GPU together (D33), so for the report that process is closed, llama-server runs alone, and the speech models
are loaded again afterwards. The page is plain HTML in demo/static/, polling this server.

"Test recording" plays a test conversation (out loud) and feeds it in as if it came from the microphone: for
rehearsing, and as a fallback on the day.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from whospoke.audio import SR, load, resample  # noqa: E402  (numpy + soundfile only: no torch in this process)
from whospoke.paths import DATA, SYNTH  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
LANGUAGES = {"hi": "Hindi + English", "kn": "Kannada"}
TEST_AUDIO = SYNTH / "test" / "test_g02_2spk_ovl-high_market5" / "mixture.wav"   # the flash-talk demo conversation
TEMPLATE = PROJECT / "src" / "whospoke" / "resources" / "airavata_chat_template.jinja"
MIN_AUDIO_S = 3.0          # first pass once this much audio exists
REFRESH_S = 1.5            # then whenever this much new audio has arrived (and the last pass is done)
ENV_BIN_S = 0.05           # waveform resolution sent to the page
LLM_LOG = Path(os.getenv("TEMP", "/tmp")) / "whospoke_live_llm.log"   # llama-server's output, for when it fails
REPORT_STEPS = ["Free the GPU for the LLM", "Start Airavata 7B (4-bit)", "Write the report", "Reload the speech models"]


# ====================================================================== engine (child process)
def engine_main(conn, language: str) -> None:
    """Holds the speech models on the GPU and answers ("run", (wav, people)) | ("voices", None) |
    ("language", code) | ("quit", None)."""
    sys.path.insert(0, str(PROJECT / "src"))
    try:
        from whospoke.__main__ import tuned_diarizer
        from whospoke.asr_backends import IndicConformerASR
        from whospoke.diarization import embed
        from whospoke.pipeline import Pipeline

        def asr_for(lang: str):
            return IndicConformerASR(decoding="rnnt", language=lang)   # the pipeline's default decoding

        pipe = Pipeline("B", diarizer=known_count(tuned_diarizer("B", "spectral")), asr_model=asr_for(language))
        pipe.warmup()
        conn.send(("ready", language))
        last = None
        while True:
            kind, arg = conn.recv()
            if kind == "quit":
                break
            try:
                if kind == "language":
                    pipe.asr.close()
                    pipe.asr = asr_for(arg)
                    pipe.warmup()
                    conn.send(("ready", arg))
                elif kind == "run":
                    wav, people = arg
                    t0 = time.perf_counter()
                    res = pipe.run(wav, n_speakers=people)
                    last = (wav, res)
                    conn.send(("result", _payload(pipe, res, time.perf_counter() - t0)))
                elif kind == "voices":
                    conn.send(("voices", _voices(pipe, *last, embed) if last else {}))
            except Exception as exc:                       # keep the engine alive; the page shows the message
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except Exception:
        conn.send(("fatal", traceback.format_exc()))


def known_count(diarizer):
    """When the number of people is given: keep the automatic clustering if it found that many speakers, else
    cluster again with the count fixed. On the 72 test conversations this lowers DER from 20.3 % (count found
    automatically, as published) to 14.8 % (scripts/eval_speaker_count.py). With no count given it is unchanged."""
    auto_assign = diarizer.assign

    def assign(features, n_speakers=None):
        auto = auto_assign(features)
        if n_speakers is None or len(auto.speakers) == n_speakers:
            return auto
        return auto_assign(features, n_speakers)

    diarizer.assign = assign
    return diarizer


def _payload(pipe, res, seconds: float) -> dict:
    diar = res.diarization
    return {
        "duration": res.duration_s,
        "speakers": diar.speakers,
        "turns": [[round(t.start, 2), round(t.end, 2), t.speaker] for t in diar.turns],
        "overlaps": [[round(a, 2), round(b, 2)] for a, b in pipe._overlap_regions(diar)],
        "lines": [{"speaker": ln.speaker, "start": ln.start, "end": ln.end, "text": ln.text, "latin": ln.hinglish,
                   "separated": ln.audio == "separated"}
                  for ln in sorted(res.lines, key=lambda x: x.start) if ln.text],
        "timings": {k: round(v, 3) for k, v in res.timings.items()},
        "seconds": round(seconds, 2),
    }


def _voices(pipe, wav: np.ndarray, res, embed) -> dict[str, np.ndarray]:
    """The whole recording through the separator, each track named after the speaker it sounds like (as on slide 1)."""
    diar = res.diarization
    talk: dict[str, float] = {}
    for t in diar.turns:
        talk[t.speaker] = talk.get(t.speaker, 0.0) + t.duration
    top = sorted(talk, key=talk.get, reverse=True)[:2]          # Conv-TasNet gives two tracks: the two main voices
    if len(top) < 2 or pipe.separator is None:
        return {}
    cents = pipe._centroids(wav, diar.turns, pipe._overlap_regions(diar), diar)
    if not all(s in cents for s in top):
        return {}
    est = pipe.separator.separate(wav)
    dur = len(wav) / SR

    def fingerprint(x: np.ndarray) -> np.ndarray:
        wins = [(a, min(a + 10.0, dur)) for a in np.arange(0.0, dur, 10.0) if min(a + 10.0, dur) - a > 1.0]
        v = embed(x, wins).mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    fp = np.stack([fingerprint(est[k]) for k in range(2)])
    score = fp @ np.stack([cents[s] for s in top]).T           # track x speaker
    k = 0 if score[0, 0] + score[1, 1] >= score[0, 1] + score[1, 0] else 1
    return {top[0]: _level(est[k]), top[1]: _level(est[1 - k])}


def _level(x: np.ndarray) -> np.ndarray:
    return (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32)


# ====================================================================== the hub (this process)
class Hub:
    """Microphone, take state and the engine, shared by the HTTP handlers and one worker thread."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.language = "hi"
        self.people: int | None = 2               # how many people are talking (None: count them automatically)
        self.english = not args.no_english        # English line under each transcript line (demo/indictrans.py)
        self.translations: dict[tuple[str, str], str] = {}
        self.translator_name, self.translator_note = None, "loading"
        self.status, self.message, self.error = "loading", "Loading the speech models", None
        self.engine_ready = False
        self.pending_language: str | None = None
        self.test_wav = load(args.test_audio) if args.test_audio and Path(args.test_audio).exists() else None
        self.gguf, self.llama = self._find_llm()
        self.take = 0
        self._new_take()
        self.conn = self.proc = None
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._translate_worker, daemon=True).start()

    # ------------------------------------------------------------------ take state
    def _new_take(self) -> None:
        self.take += 1
        self.recording, self.source, self.stream = False, "mic", None
        self.chunks: list[np.ndarray] = []
        self.capture_sr, self.n = SR, 0
        self.level, self.env = 0.0, []
        self._bin_peak, self._bin_fill = 0.0, 0
        self.snap_n = 0                                  # captured samples covered by the last pass
        self.result, self.result_version, self.updated_at = None, 0, None
        self.runs, self.final_pending, self.stop_requested = 0, False, False
        self.voices: dict[str, bytes] = {}
        self.report = self.report_error = None
        self.report_state, self.report_step, self.report_t0 = "idle", -1, None

    def _append(self, block: np.ndarray) -> None:
        """Audio from the mic or the test recording (capture rate, mono)."""
        with self.lock:
            if not self.recording:
                return
            self.chunks.append(block.copy())
            self.n += len(block)
            self.level = float(np.abs(block).max()) if len(block) else 0.0
            per_bin = int(ENV_BIN_S * self.capture_sr)
            i = 0
            while i < len(block):
                take = min(per_bin - self._bin_fill, len(block) - i)
                self._bin_peak = max(self._bin_peak, float(np.abs(block[i:i + take]).max()))
                self._bin_fill += take
                i += take
                if self._bin_fill >= per_bin:
                    self.env.append(round(self._bin_peak, 4))
                    self._bin_peak, self._bin_fill = 0.0, 0
            if self.n / self.capture_sr >= self.args.max_seconds:
                self.stop_requested = True

    def _audio16k(self, n: int | None = None) -> np.ndarray:
        with self.lock:
            x = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
            sr = self.capture_sr
        x = x[:n] if n else x
        x = resample(x, sr, SR)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        return (x * (0.9 / peak) if peak > 0 else x).astype(np.float32)   # one gain for the whole take

    # ------------------------------------------------------------------ actions (from the page)
    def start(self, source: str = "mic", audible: bool = True) -> tuple[bool, str]:
        with self.lock:
            if self.recording:
                return False, "Already recording."
            if not self.engine_ready or self.status in ("report", "loading"):
                return False, "Wait until the speech models are loaded."
            if source == "test" and self.test_wav is None:
                return False, "No test recording found (build the test set or pass --test-audio)."
            self._new_take()
            self.source = source
            try:
                self.stream = self._open_test(audible and not self.args.silent_test) if source == "test" \
                    else self._open_mic()
            except Exception as exc:
                return False, f"Could not open the {'speakers' if source == 'test' else 'microphone'}: {exc}"
            self.recording = True
            self.status, self.message, self.error = "recording", "Listening", None
            return True, "Recording."

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if not self.recording:
                return False, "Not recording."
            self.recording, self.level = False, 0.0
            stream, self.stream = self.stream, None
            self.final_pending = self.n / self.capture_sr >= 1.0
            self.status = "finishing" if self.final_pending else "ready"
            self.message = "Final pass on the whole recording" if self.final_pending else "Too short to process"
        if stream is not None:
            try:
                stream.stop(), stream.close()
            except Exception:
                pass
        return True, "Stopped."

    def reset(self) -> tuple[bool, str]:
        if self.recording:
            self.stop()
        with self.lock:
            if self.status in ("finishing", "report"):
                return False, "Wait for the current step to finish."
            self._new_take()
            self.status, self.message = ("ready", "Ready") if self.engine_ready else (self.status, self.message)
        return True, "Cleared."

    def set_language(self, language: str) -> tuple[bool, str]:
        if language not in LANGUAGES:
            return False, f"Unknown language {language!r}."
        with self.lock:
            if self.recording or self.status in ("finishing", "report"):
                return False, "Change the language between takes."
            if language != self.language:
                self.language, self.pending_language = language, language
                self.engine_ready = False
                self.status, self.message = "loading", f"Switching the recogniser to {LANGUAGES[language]}"
        return True, LANGUAGES[language]

    def set_people(self, people) -> tuple[bool, str]:
        people = None if people in (None, "auto", 0) else int(people)
        if people is not None and not 1 <= people <= 4:
            return False, "Choose 1 to 4 people, or automatic."
        with self.lock:
            if self.recording or self.status in ("finishing", "report"):
                return False, "Change this between takes."
            self.people = people
        return True, "Automatic" if people is None else f"{people} people"

    def set_english(self, on) -> tuple[bool, str]:
        with self.lock:
            self.english = bool(on)
            self.result_version += 1
        return True, "English line on" if on else "English line off"

    def _translate_worker(self) -> None:
        """English for new transcript lines, on the CPU (Stages 1-3 fill the GPU) and off the pipeline's critical
        path: a line appears at once and gets its English a few seconds later. Newest lines first, three at a time."""
        translator = None
        while True:
            with self.lock:
                want = self.english
                lines = list(self.result["lines"]) if self.result else []
                lang = self.language
            if not want:
                time.sleep(0.2)
                continue
            if translator is None:
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import torch
                    from indictrans import load_translator

                    torch.set_num_threads(4)              # leave the other cores to Stages 1-3
                    translator = load_translator("cpu")
                    with self.lock:
                        self.translator_name, self.translator_note = translator.name, "on"
                        self.result_version += 1
                except Exception as exc:
                    with self.lock:
                        self.translator_note = f"English line unavailable: {type(exc).__name__}: {str(exc)[:160]}"
                        self.result_version += 1
                    return
            with self.lock:                     # only settled lines: the newest ones still change from pass to pass
                end = self.result["duration"] - 3.0 if self.result and self.recording else float("inf")
            todo = [x["text"] for x in reversed(lines)
                    if x["end"] <= end and (lang, x["text"]) not in self.translations][:3]
            if not todo:
                time.sleep(0.15)
                continue
            try:
                done = translator(todo, lang)
                with self.lock:
                    self.translations.update({(lang, t): e for t, e in zip(todo, done)})
                    self.result_version += 1
            except Exception as exc:
                with self.lock:
                    self.translator_note = f"Translation failed: {type(exc).__name__}: {str(exc)[:160]}"
                time.sleep(1.0)

    def request_report(self) -> tuple[bool, str]:
        with self.lock:
            ok, why = self._report_available()
            if not ok:
                return False, why
            if self.recording or self.status != "done":
                return False, "Stop the take and wait for the final pass first."
            self.status, self.report_state, self.report_step = "report", "running", 0
            self.report_t0, self.report, self.report_error = time.time(), None, None
            self.message = REPORT_STEPS[0]
        return True, "Writing the report."

    def _report_available(self) -> tuple[bool, str]:
        if self.language != "hi":
            return False, "The report model (Airavata) reads Hindi only."
        if self.gguf is None:
            return False, "LLM model file not found (pass --gguf)."
        if self.llama is None:
            return False, "llama-server not found (pass --llama-bin)."
        if self.result is None or not self.result["lines"]:
            return False, "Record a conversation first."
        return True, ""

    # ------------------------------------------------------------------ audio devices
    def _open_mic(self):
        import sounddevice as sd

        dev = self.args.mic
        last = None
        for rate in (SR, None):                           # 16 kHz if the device allows it, else its own rate
            try:
                rate = rate or int(sd.query_devices(dev, "input")["default_samplerate"])
                st = sd.InputStream(device=dev, channels=1, samplerate=rate, dtype="float32",
                                    callback=lambda data, frames, t, status: self._append(data[:, 0]))
                self.capture_sr = rate
                st.start()
                return st
            except Exception as exc:
                last = exc
        raise RuntimeError(last)

    def _open_test(self, audible: bool):
        import sounddevice as sd

        wav, pos = self.test_wav, [0]
        self.capture_sr = SR

        def cb(out, frames, t, status):
            chunk = wav[pos[0]: pos[0] + frames]
            pos[0] += frames
            out[:] = 0
            if audible:
                out[: len(chunk), 0] = chunk
            self._append(chunk)
            if len(chunk) < frames:
                self.stop_requested = True
                raise sd.CallbackStop

        st = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", callback=cb)
        st.start()
        return st

    # ------------------------------------------------------------------ engine process
    def _start_engine(self, language: str) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=engine_main, args=(child, language), daemon=True)
        self.proc.start()
        self._expect("ready")
        with self.lock:
            self.engine_ready = True

    def _stop_engine(self) -> None:
        with self.lock:
            self.engine_ready = False
        if self.proc is not None:
            try:
                self.conn.send(("quit", None))
                self.proc.join(15)
            except Exception:
                pass
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(5)
        self.proc = self.conn = None

    def _ask(self, kind: str, arg=None):
        self.conn.send((kind, arg))
        return self._expect({"run": "result", "voices": "voices", "language": "ready"}[kind])

    def _expect(self, kind: str):
        got, value = self.conn.recv()
        if got == "fatal":
            raise RuntimeError(value.strip().splitlines()[-1])
        if got == "error":
            raise RuntimeError(value)
        if got != kind:
            raise RuntimeError(f"engine replied {got!r}, expected {kind!r}")
        return value

    # ------------------------------------------------------------------ worker thread
    def _worker(self) -> None:
        try:
            self._start_engine(self.language)
            with self.lock:
                self.status, self.message = "ready", "Ready"
        except Exception as exc:
            self._fail(f"The speech models did not load: {exc}")
            return
        while True:
            try:
                if not self._step():
                    time.sleep(0.05)
            except Exception as exc:
                self._fail(str(exc))
                if self.proc is not None and not self.proc.is_alive():      # the engine died: start a new one
                    with self.lock:
                        self.engine_ready, self.status, self.message = False, "loading", "Restarting the speech models"
                    try:
                        self._start_engine(self.language)
                        with self.lock:
                            self.status, self.message = "ready", "Ready"
                    except Exception as exc2:
                        self._fail(f"The speech models did not restart: {exc2}")
                time.sleep(0.5)

    def _fail(self, msg: str) -> None:
        with self.lock:
            self.error, self.message = msg, msg
            if self.status in ("finishing", "loading", "report"):
                self.status = "ready" if self.engine_ready else "error"
            if self.report_state == "running":
                self.report_state, self.report_error = "error", msg

    def _step(self) -> bool:
        """Do the next piece of work, if any. Returns False when idle."""
        if self.stop_requested and self.recording:
            self.stop()
        with self.lock:
            lang, self.pending_language = self.pending_language, None
            want_report = self.status == "report" and self.report_state == "running"
            final = self.final_pending
            live = (self.recording and self.n / self.capture_sr >= MIN_AUDIO_S
                    and (self.n - self.snap_n) / self.capture_sr >= REFRESH_S)
            take = self.take
        if lang:
            self._ask("language", lang)
            with self.lock:
                self.engine_ready = True
                self.status, self.message = "ready", f"Ready: {LANGUAGES[lang]}"
            return True
        if want_report:
            self._write_report()
            return True
        if final:
            self._run(None, take)
            with self.lock:
                self.final_pending = False
                self.message = "Pulling the two voices apart"
            voices = self._ask("voices")
            with self.lock:
                if take == self.take:
                    self.voices = {spk: _wav_bytes(x) for spk, x in voices.items()}
                    self.voices["original"] = _wav_bytes(self._audio16k())
                    self.status, self.message = "done", "Done"
            return True
        if live:
            with self.lock:
                n = self.n
            self._run(n, take)
            return True
        return False

    def _run(self, n: int | None, take: int) -> None:
        with self.lock:
            n = self.n if n is None else n
            self.snap_n = n
        res = self._ask("run", (self._audio16k(n), self.people))
        with self.lock:
            if take != self.take:                       # a reset happened meanwhile
                return
            self.result, self.updated_at = res, time.time()
            self.result_version += 1
            self.runs += 1

    # ------------------------------------------------------------------ Stage 4
    def _write_report(self) -> None:
        from whospoke.llm_postprocess import PostProcessor

        def step(i: int) -> None:
            with self.lock:
                self.report_step, self.message = i, REPORT_STEPS[i]

        server = log = None
        try:
            step(0)
            self._stop_engine()
            step(1)
            log = open(LLM_LOG, "wb")
            server = subprocess.Popen(self._llm_cmd(), stdout=log, stderr=subprocess.STDOUT)
            pp = PostProcessor()
            self._wait_for_llm(server, pp)
            step(2)
            res = self.result
            rep = pp.process_transcript({"order": "B", "duration_s": res["duration"], "lines": [
                {"speaker": x["speaker"], "start": x["start"], "end": x["end"], "text": x["text"], "hinglish": x["latin"]}
                for x in res["lines"]]}).to_dict()
            with self.lock:
                self.report = {k: rep[k] for k in ("title", "topic", "executive_summary", "key_points", "keywords",
                                                    "action_items")}
                self.report["dialogue"] = [{"speaker": d["speaker"], "start": d["start"], "english": d["translation"],
                                            "uncertain": d["uncertain"]} for d in rep["cleaned_dialogue"]]
                self.report["seconds"] = round(time.time() - self.report_t0, 1)
                self.report_state = "done"
                self.result_version += 1
        except Exception as exc:
            with self.lock:
                self.report_state, self.report_error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(15)
                except subprocess.TimeoutExpired:
                    server.kill()
            if log is not None:
                log.close()
            step(3)
            try:
                self._start_engine(self.language)
                with self.lock:
                    self.status = "done"
                    self.message = "Report written" if self.report_state == "done" else "The report failed"
            except Exception as exc:
                self._fail(f"The speech models did not reload: {exc}")

    @staticmethod
    def _wait_for_llm(server: subprocess.Popen, pp, wait_s: float = 240) -> None:
        """llama-server refuses connections until it listens, then answers 503 while loading the model."""
        from whospoke.llm_postprocess import LLMConnectionError

        deadline = time.monotonic() + wait_s
        while True:
            if server.poll() is not None:
                tail = LLM_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
                raise RuntimeError(f"llama-server stopped (exit {server.returncode}): {' | '.join(tail)}")
            try:
                pp.backend.ping(wait_s=max(1.0, deadline - time.monotonic()))
                return
            except LLMConnectionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)

    def _llm_cmd(self) -> list[str]:
        # The same server command as scripts/serve_llm.py, with the whole model on the GPU (D33).
        return [str(self.llama), "-m", str(self.gguf), "--host", "127.0.0.1", "--port", "8080", "-np", "1",
                "-c", "4096", "-ngl", str(self.args.gpu_layers), "--jinja", "--chat-template-file", str(TEMPLATE),
                "--alias", "ai4bharat/Airavata", "--cache-ram", "0"]

    def _find_llm(self) -> tuple[Path | None, Path | None]:
        cands = [self.args.gguf, os.getenv("WHOSPOKE_GGUF"), PROJECT / "models" / "airavata-q4_k_m.gguf"]
        gguf = next((Path(c) for c in cands if c and Path(c).is_file()), None)
        exe = "llama-server" + (".exe" if os.name == "nt" else "")
        bins = [self.args.llama_bin, os.getenv("WHOSPOKE_LLAMA_BIN"), shutil.which("llama-server"),
                *sorted(DATA.glob("llama-*"))]
        llama = None
        for b in bins:
            if not b:
                continue
            p = Path(b)
            p = p / exe if p.is_dir() else p
            if p.is_file():
                llama = p
                break
        return gguf, llama

    # ------------------------------------------------------------------ what the page reads
    def state(self, env_from: int, take: int) -> dict:
        with self.lock:
            if take != self.take:
                env_from = 0
            ok, why = self._report_available()
            elapsed = self.n / self.capture_sr
            return {
                "take": self.take, "status": self.status, "message": self.message, "error": self.error,
                "language": self.language, "languages": LANGUAGES, "source": self.source, "people": self.people,
                "english": self.english,
                "recording": self.recording, "elapsed": round(elapsed, 2), "max_seconds": self.args.max_seconds,
                "level": round(self.level, 4), "env_from": env_from, "env": self.env[env_from:],
                "env_bin_s": ENV_BIN_S, "result_version": self.result_version, "runs": self.runs,
                "updated_ago": round(time.time() - self.updated_at, 1) if self.updated_at else None,
                "test_available": self.test_wav is not None,
                "voices": sorted(k for k in self.voices if k != "original") + (["original"] if "original" in self.voices else []),
                "report": {"available": ok, "reason": why, "state": self.report_state, "step": self.report_step,
                           "steps": REPORT_STEPS, "error": self.report_error,
                           "elapsed": round(time.time() - self.report_t0, 1) if self.report_t0 else None},
            }

    def result_payload(self) -> dict:
        with self.lock:
            res = self.result
            if res is not None and self.english:
                res = {**res, "lines": [{**x, "english": self.translations.get((self.language, x["text"]), "")}
                                        for x in res["lines"]]}
            return {"take": self.take, "version": self.result_version, "result": res, "report": self.report,
                    "translator": self.translator_name if self.english else None,
                    "translator_note": self.translator_note if self.english else "off"}

    def audio(self, name: str) -> bytes | None:
        with self.lock:
            return self.voices.get(name)


def _wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ====================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    hub: Hub

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path.startswith("/fonts/"):
            f = STATIC / "fonts" / Path(u.path).name
            return self._send(200, f.read_bytes(), "font/ttf") if f.is_file() else self._send(404, b"", "text/plain")
        if u.path == "/api/state":
            return self._json(self.hub.state(int(q.get("env", ["0"])[0]), int(q.get("take", ["0"])[0])))
        if u.path == "/api/result":
            return self._json(self.hub.result_payload())
        if u.path.startswith("/api/audio/"):
            data = self.hub.audio(Path(u.path).stem)
            return self._send(200, data, "audio/wav") if data else self._send(404, b"", "text/plain")
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        hub = self.hub
        actions = {"/api/start": lambda: hub.start(body.get("source", "mic"), bool(body.get("audible", True))),
                   "/api/stop": hub.stop, "/api/reset": hub.reset, "/api/report": hub.request_report,
                   "/api/language": lambda: hub.set_language(body.get("language", "")),
                   "/api/people": lambda: hub.set_people(body.get("people")),
                   "/api/english": lambda: hub.set_english(body.get("on", True))}
        act = actions.get(urlparse(self.path).path)
        if act is None:
            return self._json({"ok": False, "message": "unknown action"}, 404)
        ok, msg = act()
        self._json({"ok": ok, "message": msg}, 200 if ok else 409)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mic", default=None, help="input device number or name (default: the system's default mic)")
    ap.add_argument("--list-mics", action="store_true", help="list input devices and exit")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="a take stops by itself after this long")
    ap.add_argument("--test-audio", default=str(TEST_AUDIO), help="recording used by 'Test recording'")
    ap.add_argument("--silent-test", action="store_true", help="feed the test recording without playing it aloud")
    ap.add_argument("--gguf", default=None, help="4-bit Airavata model file (default: models/airavata-q4_k_m.gguf)")
    ap.add_argument("--llama-bin", default=None, help="llama-server executable or its folder")
    ap.add_argument("--gpu-layers", type=int, default=99, help="LLM layers on the GPU (99 = all, as in D33)")
    ap.add_argument("--no-english", action="store_true", help="start with the IndicTrans2 English line switched off")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--allow-downloads", action="store_true",
                    help="let the models check Hugging Face for updates (default: cached models only, no internet)")
    args = ap.parse_args()
    if not args.allow_downloads:          # the venue may have no internet; every model is already in the local cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.list_mics:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3d}  {d['name']}  ({sd.query_hostapis(d['hostapi'])['name']})")
        return
    if args.mic is not None and str(args.mic).isdigit():
        args.mic = int(args.mic)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    Handler.hub = Hub(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"whospoke live: {url}  (speech models loading; Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.hub._stop_engine()


if __name__ == "__main__":
    main()
