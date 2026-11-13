"""Fine-tune Conv-TasNet on Hindi two-speaker mixtures with our noise (improvement A3). Needs a GPU (Colab T4 is enough).

Starts from the released checkpoint (``convtasnet`` = JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k: English read speech,
WHAM! noise) and keeps training it on mixtures made like our overlap regions:

* **Speakers:** IndicVoices Hindi *train* shard(s), the pool the dev set was drawn from (``build_dataset.py``), minus
  every speaker of the dev and test conversations and every speaker of the *valid* split (the test pool). So no
  training voice is ever heard in dev or test.
* **Noise:** the dev halves of the village and market soundscapes (``NoiseBank(..., "dev")``: first half of each
  DEMAND recording, ESC-50 folds 4–5). The test halves are never used.
* **Segments:** 3 s (as the checkpoint was trained). Half fully overlapped (like Libri2Mix "min"), half partly
  overlapped, with one person talking alone before or after, as in an overlap region with its context.
* **Loss:** negative SI-SDR with permutation-invariant training, as the checkpoint was trained.

Model selection uses the **dev** overlap regions only, scored exactly as ``scripts/eval_separation.py`` scores them
(SI-SDR improvement, 1 s context). Step 0 is the released model, so the log starts at its dev score (6.43 dB). The test
set is never touched by this script.

    python scripts/finetune_convtasnet.py                      # defaults below; resumes if interrupted
    python scripts/finetune_convtasnet.py --extra-shards 2     # also download 2 more IndicVoices train shards

Outputs: models/convtasnet_hindi.pth (best on dev; ``Separator("convtasnet-hindi")`` loads it),
         models/convtasnet_hindi_last.pt (to resume), results/finetune_convtasnet/{config.json, log.csv}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_separation import load_conv, two_speaker_overlaps  # noqa: E402
from whospoke.audio import SR, load, rms  # noqa: E402
from whospoke.corpora import fetch, read_indicvoices  # noqa: E402
from whospoke.metrics import pit_si_sdr, si_sdr  # noqa: E402
from whospoke.noise import NoiseBank  # noqa: E402
from whospoke.paths import RAW, SYNTH, conversation  # noqa: E402
from whospoke.separation import LOCAL_SEPARATORS, SEPARATORS  # noqa: E402
from whospoke.synth import trim_silence  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "finetune_convtasnet"
BEST = PROJECT / LOCAL_SEPARATORS["convtasnet-hindi"]
LAST = BEST.with_name("convtasnet_hindi_last.pt")


# ------------------------------------------------------------------ data
def held_out_speakers() -> set[str]:
    """Every speaker who appears in a dev or test conversation, plus the whole valid split (the test pool)."""
    out: set[str] = set()
    for split in ("dev", "test"):
        for p in pd.read_csv(SYNTH / split / "index.csv").path:
            meta = json.loads((conversation(p) / "reference.json").read_text(encoding="utf-8"))
            out |= set(meta["speaker_ids"].values())
    valid = RAW / "indicvoices" / "hindi" / "valid-00000-of-00001.parquet"
    if not valid.exists():                                   # e.g. the light Colab bundle: download it there
        valid = fetch("ai4bharat/IndicVoices", "hindi/valid-00000-of-00001.parquet", "indicvoices")
    out |= set(read_indicvoices(valid).speaker_id)
    return out


def training_pool(n_shards: int, exclude: set[str], max_hours: float) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Decoded utterances (int16, silence trimmed) of speakers not in ``exclude``; and their speaker index."""
    wavs, spk, names, info = [], [], {}, {"shards": [], "utterances": 0, "hours": 0.0}
    total = 0.0
    for k in range(n_shards):
        name = f"hindi/train-{k:05d}-of-00082.parquet"
        path = RAW / "indicvoices" / name
        if not path.exists():
            path = fetch("ai4bharat/IndicVoices", name, "indicvoices")
        df = read_indicvoices(path, with_audio=True)
        df = df[~df.speaker_id.isin(exclude) & (df.duration >= 2.0)]
        info["shards"].append(name)
        for r in df.itertuples():
            w = trim_silence(load(r.audio_bytes))
            if len(w) < 2 * SR:
                continue
            peak = float(np.max(np.abs(w))) or 1.0
            wavs.append(np.round(w / peak * 32000).astype(np.int16))
            spk.append(names.setdefault(r.speaker_id, len(names)))
            total += len(w) / SR / 3600
            if total >= max_hours:
                break
        print(f"  {name}: {len(wavs)} utterances, {len(names)} speakers, {total:.1f} h so far", flush=True)
        if total >= max_hours:
            break
    info.update(utterances=len(wavs), speakers=len(names), hours=round(total, 2))
    return wavs, np.asarray(spk), info


class MixtureMaker:
    """Random 3 s training mixtures: two different speakers, our dev noise, targets = the clean voices."""

    def __init__(self, wavs, spk, seg_s=3.0, p_full=0.5, p_noise=0.75, snr=(0.0, 15.0), rel_db=(-5.0, 5.0), seed=0):
        self.wavs, self.spk = wavs, spk
        self.n = int(seg_s * SR)
        self.p_full, self.p_noise, self.snr, self.rel_db = p_full, p_noise, snr, rel_db
        self.rng = np.random.default_rng(seed)
        self.noise = [NoiseBank("village", "dev"), NoiseBank("market", "dev")]

    def _crop(self, length: int) -> tuple[np.ndarray, int]:
        i = int(self.rng.integers(len(self.wavs)))
        w = self.wavs[i].astype(np.float32) / 32768.0
        if len(w) <= length:
            return np.pad(w, (0, length - len(w))), int(self.spk[i])
        a = int(self.rng.integers(len(w) - length))
        return w[a: a + length], int(self.spk[i])

    def example(self) -> tuple[np.ndarray, np.ndarray]:
        n, rng = self.n, self.rng
        while True:
            s1, k1 = self._crop(n)
            s2 = np.zeros(n, np.float32)
            if rng.random() < self.p_full:                          # both talk the whole time
                part, k2 = self._crop(n)
                s2[:] = part
            else:                                                    # one starts or stops part-way through
                off = int(rng.uniform(0.5, 2.0) * SR)
                part, k2 = self._crop(n - off)
                s2[off:] = part
                if rng.random() < 0.5:                               # the first speaker stops before the end
                    s1[int(rng.uniform(off / SR + 0.5, 3.0) * SR):] = 0.0
                if rng.random() < 0.5:                               # either order
                    s1, s2 = s2, s1
            if k1 != k2 and rms(s1) > 1e-3 and rms(s2) > 1e-3:
                break
        s1 *= 0.05 / rms(s1)
        s2 *= 0.05 / rms(s2) * 10 ** (rng.uniform(*self.rel_db) / 20)
        mix = s1 + s2
        if rng.random() < self.p_noise:
            noise = self.noise[int(rng.integers(2))].sample(n, rng)
            mix = mix + noise * rms(mix) / rms(noise) * 10 ** (-rng.uniform(*self.snr) / 20)
        g = 0.1 / max(rms(mix), 1e-6) * 10 ** (rng.uniform(-6, 6) / 20)   # random overall level
        return (mix * g).astype(np.float32), (np.stack([s1, s2]) * g).astype(np.float32)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*(self.example() for _ in range(size)))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ------------------------------------------------------------------ dev scoring (same regions as eval_separation.py)
def dev_regions(context: float = 1.0) -> list[dict]:
    regions = []
    for r in pd.read_csv(SYNTH / "dev" / "index.csv").itertuples():
        _, mix, srcs, segs = load_conv(conversation(r.path))
        for s, e, a, b in two_speaker_overlaps(segs):
            lo, hi = int(max(0, s - context) * SR), int(min(len(mix) / SR, e + context) * SR)
            x, refs = mix[lo:hi], [srcs[a][lo:hi], srcs[b][lo:hi]]
            regions.append({"x": x, "refs": refs, "base": float(np.mean([si_sdr(x, y) for y in refs])),
                            "noise": r.noise})
    return regions


@torch.inference_mode()
def score_dev(model, regions, device) -> dict:
    model.eval()
    gains = []
    for reg in regions:
        est = model(torch.from_numpy(reg["x"])[None].to(device))[0].float().cpu().numpy()[:, : len(reg["x"])]
        score, _ = pit_si_sdr(list(est), reg["refs"])
        gains.append((reg["noise"], score - reg["base"]))
    model.train()
    df = pd.DataFrame(gains, columns=["noise", "si_sdri"])
    out = {"dev_si_sdri": round(float(df.si_sdri.mean()), 3)}
    out.update({f"dev_{k}": round(float(v), 3) for k, v in df.groupby("noise").si_sdri.mean().items()})
    return out


# ------------------------------------------------------------------ training
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=6, help="stop after this many evaluations without a dev gain")
    ap.add_argument("--extra-shards", type=int, default=0, help="IndicVoices train shards to add beyond shard 0")
    ap.add_argument("--max-hours", type=float, default=20.0, help="cap on training speech kept in memory")
    ap.add_argument("--seed", type=int, default=407)
    ap.add_argument("--fresh", action="store_true", help="ignore a saved run and start again from the checkpoint")
    args = ap.parse_args()

    from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
    from asteroid.models import ConvTasNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU; this will be very slow (use a Colab GPU runtime)", flush=True)
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    BEST.parent.mkdir(parents=True, exist_ok=True)

    print("held-out speakers ...", flush=True)
    exclude = held_out_speakers()
    print(f"  {len(exclude)} speakers excluded (dev + test conversations, valid split)", flush=True)
    print("training pool ...", flush=True)
    wavs, spk, pool = training_pool(1 + args.extra_shards, exclude, args.max_hours)
    assert pool["speakers"] >= 20, "too few training speakers"
    maker = MixtureMaker(wavs, spk, seed=args.seed)
    print("dev regions ...", flush=True)
    regions = dev_regions()
    print(f"  {len(regions)} dev overlap regions", flush=True)

    model = ConvTasNet.from_pretrained(SEPARATORS["convtasnet"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    log_path = OUT / "log.csv"
    step, best, stale, log = 0, -np.inf, 0, []

    if LAST.exists() and not args.fresh:
        state = torch.load(LAST, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, best, stale = state["step"], state["best"], state["stale"]
        maker.rng = np.random.default_rng(args.seed + step)
        log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        print(f"resumed at step {step} (best dev SI-SDRi {best:.2f} dB)", flush=True)
    else:
        first = {"step": 0, "train_loss": np.nan, "lr": args.lr, "elapsed_s": 0.0, **score_dev(model, regions, device)}
        log.append(first)
        best = first["dev_si_sdri"]
        torch.save(model.serialize(), BEST)
        print(f"step 0 (released model): dev SI-SDRi {best:.2f} dB", flush=True)

    config = {**vars(args), "pool": pool, "excluded_speakers": len(exclude), "dev_regions": len(regions),
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "start_checkpoint": SEPARATORS["convtasnet"], "segment_s": 3.0, "noise": "village+market, dev halves",
              "loss": "PIT negative SI-SDR"}
    (OUT / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    model.train()
    t0, losses = time.perf_counter(), []
    while step < args.steps and stale < args.patience:
        x, y = maker.batch(args.batch)
        x, y = x.to(device), y.to(device)
        est = model(x)[..., : x.shape[-1]]
        loss = loss_fn(est, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss))
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            dev = score_dev(model, regions, device)
            sched.step(dev["dev_si_sdri"])
            row = {"step": step, "train_loss": round(float(np.mean(losses)), 3), "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(time.perf_counter() - t0, 1), **dev}
            log.append(row)
            losses = []
            if dev["dev_si_sdri"] > best + 0.01:
                best, stale = dev["dev_si_sdri"], 0
                torch.save(model.serialize(), BEST)
                mark = "  <- best, saved"
            else:
                stale += 1
                mark = ""
            pd.DataFrame(log).to_csv(log_path, index=False)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "stale": stale}, LAST)
            print(f"step {step}: train loss {row['train_loss']:.2f}, dev SI-SDRi {dev['dev_si_sdri']:.2f} dB{mark}",
                  flush=True)
    pd.DataFrame(log).to_csv(log_path, index=False)
    reason = "patience" if stale >= args.patience else "steps"
    print(f"done ({reason}): best dev SI-SDRi {best:.2f} dB (released model {log[0]['dev_si_sdri']:.2f} dB) -> {BEST}")


if __name__ == "__main__":
    main()
