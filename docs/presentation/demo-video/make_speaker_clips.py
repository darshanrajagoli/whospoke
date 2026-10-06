"""Render the two "separated voice" clips that sit under the demo clip on slide 1.

whospoke's separator (Milestone 1, ConvTasNet, the pipeline's default) is run on the same test recording as the demo
clip, exactly as `python -m whospoke separate` does. Each output track is cut to 28.0–35.5 s, the stretch of the demo
clip (26.4–44.8 s) where both people talk at once, so each clip is one voice pulled out of the overlap.
Each track is named after the diarized speaker whose voice it sounds most like over the whole demo window (the same
voice-fingerprint match the pipeline uses to pick a separated voice). The true voices are used only to print a check, never to choose.

Output: speaker_a.mp4, speaker_b.mp4 (992x280, 25 fps, H.264 + AAC) and a still of each (speaker_*_still.png).
Needs the project venv (models), the test audio (scripts/build_dataset.py), ffmpeg on PATH, Pillow and soundfile.
Run from the repository root:
    python docs/presentation/demo-video/make_speaker_clips.py
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from whospoke.__main__ import tuned_diarizer  # noqa: E402
from whospoke.audio import SR, load  # noqa: E402
from whospoke.diarization import embed  # noqa: E402
from whospoke.pipeline import Pipeline  # noqa: E402
from whospoke.paths import SYNTH  # noqa: E402
from whospoke.separation import Separator  # noqa: E402

CONV = SYNTH / "test/test_g02_2spk_ovl-high_market5"
FONTS = HERE / ".fonts"
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/"
for name in ("Poppins-Medium.ttf", "Poppins-SemiBold.ttf"):
    if not (FONTS / name).exists():
        FONTS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(GF + name, FONTS / name)
T0, T1 = 26.4, 44.8                      # the demo clip: used to match each track to a speaker
C0, C1 = 28.0, 35.5                      # the overlap inside it: what each speaker clip plays
FPS, W, H = 25, 992, 280
IVORY, DARK, GRAY, MID = "#FAF9F5", "#141413", "#5E5D59", "#B0AEA5"
COL = {"Speaker_A": "#D97757", "Speaker_B": "#6A9BCC"}
LABEL = ImageFont.truetype(str(FONTS / "Poppins-SemiBold.ttf"), 48)
SMALL = ImageFont.truetype(str(FONTS / "Poppins-Medium.ttf"), 40)
L, R, CY, AMP = 48, W - 48, 196, 56


def sisdr(e, r):
    e, r = e - e.mean(), r - r.mean()
    a = (e @ r) / (r @ r + 1e-9) * r
    return 10 * np.log10((a @ a) / ((e - a) @ (e - a) + 1e-9))


def tracks_by_speaker():
    wav = load(CONV / "mixture.wav")
    pipe = Pipeline("B", diarizer=tuned_diarizer("B", "spectral"), asr_model=object())
    diar = pipe.diarizer(wav, None)
    assert diar.timeline_json() == json.load(open(ROOT / "results/demo/timeline.json")), "diarization differs from results/demo"
    centroids = pipe._centroids(wav, diar.turns, pipe._overlap_regions(diar), diar)
    est = Separator("convtasnet").separate(wav)
    peak = float(np.abs(est).max())
    if peak > 0.99:                      # as `whospoke separate` does
        est = est * (0.99 / peak)
    a, b = int(C0 * SR), int(C1 * SR)
    fp = np.concatenate([embed(est[k], [(T0, T1)]) for k in range(2)])
    fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9
    score = fp @ np.stack([centroids["Speaker_A"], centroids["Speaker_B"]]).T     # track x speaker
    ka = int(np.argmax(score[:, 0] - score[:, 1]))
    out = {"Speaker_A": est[ka, a:b], "Speaker_B": est[1 - ka, a:b]}
    mix = wav[a:b]
    for k in ("S1", "S2"):
        ref = load(CONV / f"sources/{k}.wav")[a:b]
        print(f"check vs true voice {k}: mixture {sisdr(mix, ref):+.1f} dB, "
              + ", ".join(f"{s} {sisdr(x, ref):+.1f} dB" for s, x in out.items()))
    return out


def frame(env, spk, t, playing):
    im = Image.new("RGB", (W, H), IVORY)
    d = ImageDraw.Draw(im)
    name = spk.replace("_", " ")
    if not playing:                       # still: a play glyph before the label
        d.polygon([(L, 44), (L, 100), (L + 48, 72)], fill=COL[spk])
        x0 = L + 72
    else:
        x0 = L
    d.text((x0, 72), name, font=LABEL, fill=DARK, anchor="lm")
    d.text((R, 72), "separated voice", font=SMALL, fill=GRAY, anchor="rm")
    px = L + (t - C0) / (C1 - C0) * (R - L)
    for i, e in enumerate(env):
        x = L + i * 8
        h = max(2, e * AMP)
        d.rounded_rectangle([x, CY - h, x + 4, CY + h], radius=2, fill=COL[spk] if playing and x <= px else MID)
    if playing:
        d.line([px, CY - AMP - 8, px, CY + AMP + 8], fill=DARK, width=3)
    return im


def render(spk, x):
    x = x / (np.abs(x).max() + 1e-9) * 0.9          # level each voice to the same peak
    tag = spk[-1].lower()
    wav = FONTS / f"speaker_{tag}.wav"
    sf.write(wav, x, SR)
    env = np.array([np.abs(c).max() for c in np.array_split(x, (R - L) // 8)])
    env /= env.max()
    out = HERE / f"speaker_{tag}.mp4"
    clip = FONTS / f"speaker_{tag}_48k.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af",
                    "afade=t=in:d=0.3,afade=t=out:st=%.2f:d=0.5" % (C1 - C0 - 0.5), "-ac", "1", "-ar", "48000",
                    str(clip)], check=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-i", str(clip), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
                          str(out)], stdin=subprocess.PIPE)
    for k in range(int((C1 - C0) * FPS)):
        p.stdin.write(frame(env, spk, C0 + k / FPS, True).tobytes())
    p.stdin.close(); p.wait()
    frame(env, spk, C0, False).save(HERE / f"speaker_{tag}_still.png")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    for spk, x in tracks_by_speaker().items():
        render(spk, x)
