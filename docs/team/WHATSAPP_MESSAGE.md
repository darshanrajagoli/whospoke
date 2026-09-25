# WhatsApp message to the team

Copy everything between the lines.

---

Hey guys 👋 Project update for the 6 Oct midsem review: **Milestones 1–3 are built, tested and evaluated.** It's all here 👉 https://github.com/darshanrajagoli/whospoke

*What it does:* you give it a noisy recording of people talking over each other in Hindi + English, and it tells you **who spoke, what they said, and when**. It outputs a timeline, a Devanagari transcript, a Hinglish transcript and subtitles.

*Headline results (on test audio it never saw during tuning):*
• Our order beats the proposal's: *49.5 % vs 55.8 %* who-said-what error, and *20.3 % vs 26.2 %* who-spoke-when error (better in 50 of 72 test conversations)
• Separating only the overlaps works about *2× better* than separating the whole recording (+9.6 dB vs +4.5 dB)
• Our from-scratch clustering is on par with pyannote 3.1 (difference within error bars)
• Hindi ASR: IndicConformer *21 % vs 38 %* word error for IndicWav2Vec on real-world audio; *82 %* of English words inside Hindi come out correctly spelled in Hinglish

We built the professor's order *and* an alternative, and let the numbers decide. That's the one design change we're making, and it's backed by data. Every change from his proposal is listed in `DEVIATIONS.md`, with the reason for each.

@Vismay can you own everything below and split it between you, Abhinav and Shrivaths however works best? 🙏

*1. Explainer docs* 📚
Paste this message plus the repo link into your AI, and have it read `README.md`, `docs/RESULTS.md`, `docs/PIPELINE.md`, `DECISIONS.md` and `DEVIATIONS.md`. Then build one explainer per part of the pipeline:
• Stage 1: Speech separation (Conv-TasNet)
• Stage 2: Diarization (voice fingerprints, spectral clustering vs GMM)
• Stage 3: Hindi ASR + Hinglish romanisation
• How we test it: simulated conversations, and metrics like SI-SDR, DER, WER and cpWER
• Why Order B beats the proposal's Order A
Each one needs *(a)* an ELI5 of what's happening and *why it's done this way*, and *(b)* the full proper technical explanation. These double as our prep for questions, so all of us should be able to answer anything on any stage.

*2. The presentation* 🎤
3–4 minutes, about **2 slides**, short and sweet. Make it look *really* good: clean, visual, very few words, and use the figures in `results/figures/`. A rough flow:
• Slide 1: the problem + our pipeline diagram (who-spoke-when → separate only the overlaps → Hindi ASR → Hinglish transcript)
• Slide 2: results: `order_A_vs_B.png` + `error_cascade.png`, and one example transcript from `results/demo/`
I'll take the walkthrough and the Order A vs B story on the day, since I've been deepest in the numbers. You guys own how it looks, and we'll split the Q&A by stage based on the explainers. Let's do a run-through together a day or two before.

*3. Red-team audit* 🔍
I want someone to try to break this before the prof does. Open `docs/team/RED_TEAM_AUDIT_PROMPT.md` in the repo and paste the prompt into an AI coding assistant that can read the whole repo (Claude Code / Cursor / Claude or ChatGPT with the repo zip). It will write `audit/RED_TEAM_AUDIT.md` without changing any code. Push that file to the repo (you've been added as a collaborator, so check your GitHub invites) and ping me. I'll fix whatever it finds.

Let's aim to have the explainers + slides by *Sat 3 Oct* so there's time for a run-through 🙌
