"""Build vismAI.pptx, the submitted deck, from the rendered slides and the three clips.

Each slide is a full-bleed picture of the published slide (so the deck looks the same on any PC, whatever fonts it
has), with the speaker notes from deck/slides/*.html. Slide 1 gets the three clips embedded over their stills, each
playing with sound when clicked. Also writes vismAI.pdf (stills only).

The pictures come from rendering deck/slides/*.html at 1920x1080 in a browser (slides/*.png next to this file).
Needs Windows with PowerPoint and pywin32. Run from the repository root:
    python docs/presentation/demo-video/build_pptx.py
"""
import pathlib
import re

import win32com.client

HERE = pathlib.Path(__file__).resolve().parent
SLIDES = ["intro", "results", "next"]
PT = 0.5                                   # 1920 px slide = 960 pt
CLIPS = [                                  # clip, still, (left, top, width, height) in slide px
    ("whospoke_demo.mp4", "whospoke_demo_still.png", (832, 133, 960, 540)),
    ("speaker_a.mp4", "speaker_a_still.png", (832, 767, 472, 133)),
    ("speaker_b.mp4", "speaker_b_still.png", (1320, 767, 472, 133)),
]
FADE, LAYOUT_BLANK = 1793, 12


def notes(sid):
    html = (HERE / "deck/slides" / f"{sid}.html").read_text(encoding="utf-8")
    return re.search(r"<aside>(.*?)</aside>", html, re.S).group(1).strip().replace("\n", "\r")


def main():
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Add(0)
    pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight = 960, 540
    for i, sid in enumerate(SLIDES, 1):
        s = pres.Slides.Add(i, LAYOUT_BLANK)
        s.Shapes.AddPicture(str(HERE / "slides" / f"{sid}.png"), 0, -1, 0, 0, 960, 540)
        s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes(sid)
        s.SlideShowTransition.EntryEffect = FADE
    s = pres.Slides(1)
    for clip, still, (x, y, w, h) in CLIPS:
        v = s.Shapes.AddMediaObject2(str(HERE / clip), 0, -1, x * PT, y * PT, w * PT, h * PT)
        v.MediaFormat.SetDisplayPictureFromFile(str(HERE / still))
        v.MediaFormat.Volume = 1.0
        v.Name = clip                      # PowerPoint adds its own trigger: a click on the clip plays or pauses it
    while s.TimeLine.MainSequence.Count:     # nothing plays on a slide click; only a click on a clip plays it
        s.TimeLine.MainSequence.Item(1).Delete()
    pres.SaveAs(str(HERE / "vismAI.pptx"), 24)
    pres.SaveAs(str(HERE / "vismAI.pdf"), 32)
    pres.Close()
    print(HERE / "vismAI.pptx")


if __name__ == "__main__":
    main()
