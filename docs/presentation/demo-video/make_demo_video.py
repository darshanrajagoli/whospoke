"""Render the flash-talk demo clip (whospoke_demo.mp4 + whospoke_demo_still.png).

The real test recording (test_g02_2spk_ovl-high_market5, the one in results/demo) plays while whospoke's own
output for it, the speaker timeline and the Hinglish transcript in results/demo, appears in sync. Nothing is
re-run: the clip replays the committed output against the audio. 1920x1080, 25 fps, H.264 + AAC.

Needs the test audio (rebuild with scripts/build_dataset.py), ffmpeg on PATH, Pillow and soundfile.
Run from the repository root: python docs/presentation/demo-video/make_demo_video.py
"""
import json
import pathlib
import subprocess
import urllib.request

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEMO = ROOT / "results/demo"
DATA = pathlib.Path((ROOT / "data_location.txt").read_text().strip()) if (ROOT / "data_location.txt").exists() else ROOT / "data"
WAV = str(DATA / "synth/test/test_g02_2spk_ovl-high_market5/mixture.wav")

# Poppins and Lora (SIL Open Font License), fetched once into a git-ignored folder
FONTS = HERE / ".fonts"
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
for name, path in [("Poppins-Medium.ttf", "poppins/Poppins-Medium.ttf"), ("Poppins-SemiBold.ttf", "poppins/Poppins-SemiBold.ttf"),
                   ("Lora.ttf", "lora/Lora%5Bwght%5D.ttf")]:
    if not (FONTS / name).exists():
        FONTS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(GF + path, FONTS / name)
T0, T1, FPS, W, H = 26.4, 44.8, 25, 1920, 1080

IVORY, OAT, DARK, BODY, GRAY, MID = "#FAF9F5", "#F0EEE6", "#141413", "#3D3D3A", "#5E5D59", "#B0AEA5"
COL = {"Speaker_A": "#D97757", "Speaker_B": "#6A9BCC"}
f = lambda name, size: ImageFont.truetype(str(FONTS / name), size)
EYEBROW, LABEL, SPK, TEXT, CLOCK = f("Poppins-Medium.ttf", 26), f("Poppins-Medium.ttf", 26), f("Poppins-SemiBold.ttf", 30), f("Lora.ttf", 40), f("Poppins-SemiBold.ttf", 30)

# audio + waveform envelope over the window
audio, sr = sf.read(WAV)
if audio.ndim > 1:
    audio = audio.mean(1)
seg = audio[int(T0 * sr):int(T1 * sr)]
M, L, R = 128, 348, W - 128
nb = (R - L) // 6
env = np.array([np.abs(c).max() for c in np.array_split(seg, nb)])
env = env / env.max()

timeline = [x for x in json.load(open(DEMO / "timeline.json", encoding="utf-8")) if x["end_s"] > T0 and x["start_s"] < T1]
lines = [l for l in json.load(open(DEMO / "transcript.json", encoding="utf-8"))["lines"] if T0 <= l["start"] < T1]
xof = lambda t: L + (min(max(t, T0), T1) - T0) / (T1 - T0) * (R - L)


def wrap(text, font, width):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width:
            cur = t
        else:
            out.append(cur); cur = w
    return out + [cur]


def frame(t):
    im = Image.new("RGB", (W, H), IVORY)
    d = ImageDraw.Draw(im)
    d.text((M, 96), "REAL TEST CONVERSATION  ·  2 SPEAKERS  ·  HEAVY OVERLAP  ·  MARKET NOISE", font=EYEBROW, fill=GRAY)
    s = int(t)
    d.text((R, 96), f"00:{s:02d}", font=CLOCK, fill=DARK, anchor="ra")
    # waveform
    px = xof(t)
    cy, amp = 250, 80
    for i, e in enumerate(env):
        x = L + i * 6
        h = max(2, e * amp)
        d.rounded_rectangle([x, cy - h, x + 3, cy + h], radius=1, fill=DARK if x <= px else MID)
    # speaker lanes, revealed up to the playhead
    for k, (spk, name) in enumerate((("Speaker_A", "Speaker A"), ("Speaker_B", "Speaker B"))):
        y = 370 + k * 56
        d.rounded_rectangle([L, y, R, y + 36], radius=8, fill=OAT)
        d.text((M, y + 18), name, font=LABEL, fill=DARK, anchor="lm")
        for seg_ in timeline:
            if seg_["speaker"] == spk and seg_["start_s"] < t:
                a, b = xof(seg_["start_s"]), xof(min(seg_["end_s"], t))
                if b - a > 2:
                    d.rounded_rectangle([a, y, b, y + 36], radius=8, fill=COL[spk])
    d.line([px, 150, px, 482], fill=DARK, width=3)
    d.text((M, 506), "WHAT WHOSPOKE WROTE, FROM THIS AUDIO ALONE", font=LABEL, fill=GRAY)
    # transcript: lines appear as they start; newest at the bottom, older ones fade
    shown = [l for l in lines if l["start"] <= t]
    keep, total = [], 0
    for i, l in enumerate(reversed(shown)):
        wl = wrap(l["hinglish"], TEXT, R - L)
        hgt = len(wl) * 54 + 26
        if total + hgt > 1010 - 580:
            break
        keep.append((l, wl, i)); total += hgt
    blocks, y = [], 580
    for l, wl, i in reversed(keep):
        blocks.append((l, wl, y, i)); y += len(wl) * 54 + 26
    for l, wl, yy, i in blocks:
        fade = [DARK, "#8A8985", "#B9B7B0", "#D6D4CD"][min(i, 3)]
        lab = COL[l["speaker"]] if i == 0 else fade
        d.text((M, yy + 8), l["speaker"].replace("_", " "), font=SPK, fill=lab)
        for j, ln in enumerate(wl):
            d.text((L, yy + j * 54), ln, font=TEXT, fill=fade)
    return im


if __name__ == "__main__":
    out = HERE / "whospoke_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(T0), "-t", str(T1 - T0), "-i", WAV,
                    "-af", "afade=t=in:d=0.3,afade=t=out:st=%.2f:d=0.5" % (T1 - T0 - 0.5), "-ac", "1", "-ar", "48000",
                    str(FONTS / "clip.wav")], check=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-i", str(FONTS / "clip.wav"), "-c:v", "libx264", "-preset", "slow",
                          "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                          "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = int((T1 - T0) * FPS)
    for k in range(n):
        p.stdin.write(frame(T0 + k / FPS).tobytes())
    p.stdin.close(); p.wait()
    frame(T1 - 0.05).save(HERE / "whospoke_demo_still.png")
    print(out, out.stat().st_size)
