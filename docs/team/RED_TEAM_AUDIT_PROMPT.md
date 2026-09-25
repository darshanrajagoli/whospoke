# Red-team audit — prompt to paste into your AI

**For:** Vismay · **Repo:** https://github.com/darshanrajagoli/whospoke · **Where the result goes:** `audit/RED_TEAM_AUDIT.md` in that repo.

How to use: open an AI coding assistant that can read the whole repository (Claude Code, Cursor, Copilot
Workspace, or ChatGPT/Claude with the repo zip uploaded), paste everything in the box below, and let it
run. It must not change any code — it only writes the audit file. Commit and push that file, then tell
Darshan it's up.

---

```text
You are an independent red-team reviewer for a university AI project (BITS Pilani, CS F407). Your job is to
find everything that is wrong, weak, overclaimed or fragile BEFORE the professor does. Be adversarial and
specific. Do not be polite; be correct. Do NOT modify any existing file. Your only output is one new file:
audit/RED_TEAM_AUDIT.md.

CONTEXT
- Repository: https://github.com/darshanrajagoli/whospoke (clone it). Start by reading, in this order:
  README.md, INDEX.md, Audio_Engineering_AI_Project_Proposal.docx (the professor's brief — extract its
  text), DEVIATIONS.md, DECISIONS.md, docs/PIPELINE.md, docs/RESULTS.md, then all code in src/whospoke/,
  scripts/ and tests/.
- The project is a "who spoke what and when" pipeline for noisy, overlapping, Hindi/English code-switched
  audio: Stage 1 speech separation (Conv-TasNet), Stage 2 speaker diarization (VAD + speaker embeddings +
  spectral clustering / GMM), Stage 3 Hindi ASR (IndicConformer / IndicWav2Vec) + Hinglish romanisation.
  Stage 4 (LLM post-processing) is intentionally NOT built yet (planned after the mid-semester review).
- Evaluation uses simulated conversations built from real IndicVoices speech + real background noise,
  with a dev set for tuning and a test set for reporting.

WHAT TO CHECK (go through every item; say "checked, no issue" when that is the finding)
1. Proposal compliance: for each milestone and deliverable in the proposal, is it delivered? Is every
   deviation listed in DEVIATIONS.md, and is each justification actually convincing? Anything missing?
2. Evaluation validity:
   - Data leakage: can any test speaker, utterance or noise recording influence tuning or model choice?
     (Check scripts/build_dataset.py, scripts/tune_diarization.py, scripts/eval_asr.py, noise.py splits.)
   - Pretrained-model leakage: were any pretrained models trained on the test speech? Is it disclosed?
   - Are the simulated conversations realistic enough for the claims made? What real-world effects are
     missing (reverberation, phone codecs, >2 overlapping speakers, backchannels, very short turns, far-field)?
   - Are sample sizes large enough for the differences claimed? Are there confidence intervals or paired
     comparisons? Would any conclusion flip with a different random seed?
3. Metric correctness: read src/whospoke/metrics.py line by line. Is SI-SDR right? Is DER computed with
   overlap included and a sensible collar? Is cpWER implemented correctly (speaker permutation, unmatched
   speakers, pooling)? Is text normalisation fair to both ASR models (Unicode forms, nukta, chandrabindu vs
   anusvara, punctuation, numbers)? Write small counter-examples if you suspect a bug.
4. Code correctness: look for off-by-one errors in sample/frame/time conversions, window stitching in
   separation.py, the frame-label/smoothing logic in diarization.py, the overlap assignment, the Order-A
   leakage suppression, the Order-B choice of which separated voice to keep, long-segment splitting, and
   anything that silently swallows errors.
5. Claims vs evidence: for every number and every claim in README.md, docs/RESULTS.md and the slides (if
   present), find the file that produced it and confirm it matches. Flag any claim stronger than its evidence.
6. Reproducibility: could a new person reproduce the results from README instructions alone? Missing
   steps, hidden dependencies, hard-coded paths, Windows-only assumptions, unpinned versions, gated
   Hugging Face models not mentioned.
7. Tests: which important behaviours are NOT tested? Are any tests trivially passing?
8. Hinglish romaniser (src/whospoke/hinglish.py): find words it romanises badly; check that its reported
   accuracy was measured on data not used to build its lexicon.
9. Presentation risk: list the 10 hardest questions an examiner could ask about this project, each with the
   honest best answer based on the repo.

OUTPUT FORMAT for audit/RED_TEAM_AUDIT.md
- Title, date, and a 5-line executive summary (overall verdict + the three most serious issues).
- A findings table: ID | Severity (Critical / High / Medium / Low) | Area | One-line summary | File:line.
- Then one section per finding: What is wrong · Evidence (quote code or numbers, with file:line) ·
  Why it matters · How to reproduce · Suggested fix (concrete).
- A section "Checked and found fine" listing what you verified with no issue.
- The "10 hardest examiner questions" section with answers.
Severity guide: Critical = a headline result or claim is wrong; High = a result could change or a proposal
requirement is unmet; Medium = weakens credibility/reproducibility; Low = polish.
Do not pad. Every finding must be specific enough that someone can fix it without asking you anything.
```
