"""Did VoxCPM say what it was told? -- the gate every synthetic clip passes before training.

    py adapter/verify_synth.py --selftest                                   # 13 detectors, ~1 s
    py adapter/verify_synth.py --prefetch --asr panel                       # download the jury, ~7 GB
    py adapter/verify_synth.py --manifest <m.jsonl> --clips <dir>           # cheap checks only
    py adapter/verify_synth.py --manifest <m.jsonl> --clips <dir> --asr panel   # the full panel

WHY THIS EXISTS. He observed it: "sometimes it make some random voice that not what it should
speak." That is not a misconfiguration, it is what autoregressive TTS does -- Peinl et al. 2022:
"due to their stochastic nature, the output may vary from very good to completely unusable from
one inference to another." The model samples at every step (which is what stops the voice sounding
flat); one unlucky sample lands it somewhere its training never covered, and it loops, skips, or
drifts off the text. NOTHING ERRORS. It writes a wav of the wrong thing.

THE PANEL, AND WHY IT IS SHAPED THE WAY IT IS. His call: "just download the best thai asr from each
brand include typhoon to verified the vox voice." So `--asr panel` runs the best Thai model each org
ships -- biodatlab, NECTEC Pathumma, Alibaba Qwen3-ASR, Typhoon -- plus typhoon-asr-realtime, the
model actually being trained. Each is loaded, run over every surviving clip, and unloaded; one
model is in memory at a time.

🔴 THE MODEL BEING TRAINED IS ON THE PANEL BUT NOT ON THE JURY, AND THAT IS THE WHOLE DESIGN.
Let it vote and you keep exactly the clips it ALREADY understands, discarding the hard ones -- a
filter that deletes the training signal and leaves you wondering why the arm did nothing. Same
shape as RESEARCH-FINDINGS.md F5, where including synthetic evaluation audio made every model look
~2x better. Recorded instead of counted, its opinion becomes the most useful column in the output:
clips the jury heard and it could not are the ones with something left to teach it, written to
*_verify_teach.jsonl. --let-target-vote turns that into a deliberate circularity experiment.

⚠️ FOUR BRANDS IS NOT FOUR OPINIONS. Three of the four jurors are fine-tunes of the same OpenAI
whisper-large-v3 checkpoint, so they make correlated mistakes and a unanimous verdict from three
siblings is weaker than it looks. Qwen3-ASR is the only structurally independent juror. The report
prints pairwise agreement so that correlation is visible rather than assumed away.

🔴 THE LITERATURE SAYS REGENERATE, NOT FILTER (found 2026-09-17, `H37`). Asaria et al. 2026 name
this exact failure -- "stochastic catastrophic failures: silence, terminate early, or collapse into
repetitive or hallucinated content" -- and show best-of-N ASR self-verification drives it to
NEAR-ZERO by N=2, across 4 TTS systems and 3 codecs. This file discards what it rejects, which
throws the keep-rate away; regenerating at N=2 keeps nearly all of it. The right shape is a
generate->verify->regenerate loop in colab/gen_colab.py, with this as its verifier.
⚠️ And their ceiling applies here: no amount of round-tripping teaches a TTS to say a rare name it
cannot say, which is exactly what phase2_rare asks of it.

⚠️ FALSE ALARMS ARE THE NORM, NOT THE EXCEPTION (`H38`). Lau et al. 2023 measured 21-34% of
TTS-derived ASR "failures" to be false alarms across 5 ASR and 4 TTS systems. The 2 of 12 rejected
here on 2026-09-17 by a tail check that had met a Thai spelling variant sit inside that range. Treat
every reject as suspect until an ear says otherwise -- see --calibrate-sample.

⚠️ WHAT THIS CANNOT DO. It cannot tell you the voice sounds right. colab/gen_colab.py:412 states the
repo's standing rule: "A waveform check cannot tell you the voice is right. Nothing in this repo
trusts an automated verdict on how audio sounds." This tool's job is to catch clips that say the
WRONG WORDS. Calibrating its thresholds is a job for ears -- see --calibrate-sample.

⚠️ THRESHOLDS COME FROM THE DATA, NOT FROM ME. Every duration/ratio bound is a percentile of this
corpus's own distribution, because VoxCPM's own `retry_badcase_ratio_threshold = 6.0` is a default
that was not calibrated on Thai and Thai tokenises differently. The script prints the distributions
it derived so a reader can disagree with them.

CER HERE IS A GATE, NOT A PUBLISHED NUMBER. It is plain Levenshtein over adapter/_thai.py's
`for_scoring` (SARA AM folded on BOTH sides -- F1: 26.6% of mean CER is that fold). Arm results go
through adapter/_score.py, which pins the alignment backend; this does not, and must never be
quoted as an arm number.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import (  # noqa: E402  -- the one Thai normalisation definition
    DIGIT_RUN as _DIGIT_RUN,
    for_scoring,
    infix_cer,
    thai_cardinal,
    thai_digitwise,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"

SR = 16000
THAI = re.compile(r"[฀-๿]")
# What a Thai clip's transcript may legitimately contain. Anything else means the model left Thai,
# and that clip is unusable at any CER (the rule verify.py on the endpointer line already applied).
ALLOWED = re.compile(r"[฀-๿\s\.,!\?;:\-'\"()\[\]]|[A-Za-z0-9]")

# VoxCPM's own bad-case rule (voxcpm/model/voxcpm.py:479) is
#     audio_feature_frames >= len(text_tokenizer(text)) * 6.0   ->  badcase, retry
# and max_len is that same expression + 10, so with retry OFF a runaway is truncated there.
#
# 🔴 THIS SCRIPT DELIBERATELY DOES NOT REIMPLEMENT THAT TEST. `target_text_length` is VoxCPM's own
# TOKENIZER length, not a character count, and the frame duration depends on its audio VAE and
# patch_size. Approximating either turns a threshold into a guess wearing a decimal point -- the
# exact thing this file argues against elsewhere. The data-driven chars/second bounds below catch
# both run-ons and truncations without needing the constants.
#
# To calibrate 6.0 for Thai properly, log the ratio with VoxCPM's OWN tokenizer at generation time,
# in colab/gen_colab.py where the package is importable. That is a one-line addition there and it
# is what Phase 0a of the design actually asks for.
VOXCPM_RATIO_DEFAULT = None


# --------------------------------------------------------------------------------------- utilities

def levenshtein(a: str, b: str) -> int:
    """Edit distance. Hand-rolled so no dependency can move a number -- README rule 5."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Strict CER. SARA AM folded both sides, whitespace and punctuation dropped."""
    r, h = for_scoring(ref), for_scoring(hyp)
    return 1.0 if not r else levenshtein(r, h) / len(r)


# ------------------------------------------------------- convention, which is not error
#
# 🔴 MEASURED 2026-09-16, ON THE FIRST TWO REAL CLIPS THIS PANEL EVER SAW. biodatlab transcribed
# both essentially correctly and scored CER 0.232 and 0.396, because:
#
#   reference  ...ตลาด Khlong Toei เลย...      hyp  ...ตลาดคลองเตยเลย...       (script)
#   reference  ...อีกสองคืน...จองบีทู...        hyp  ...อีก 2 คืน...จอง B2...   (digits, script)
#
# Both hypotheses are RIGHT. A gate at --cer-max 0.10 rejects both. On synthetic clips that means
# discarding good audio because the verifier writes numbers and names in a different convention
# from the label -- and it would bite hardest on the rare-entity bank, which is the part that
# matters. This is F1 and F2 arriving in a third place: three invisible text conventions (SARA AM,
# digits, script), each able to produce a confident wrong answer.
#
# So the GATE runs on a convention-tolerant CER, and the strict one is still recorded, because the
# gap between them is itself informative.

# The two number spellers and the digit-run regex moved to `_thai.py` on 2026-09-17,
# when `_ortho.py` became a second caller. Same code, one home.


def cer_tolerant(ref: str, hyp: str, max_variants: int = 8) -> float:
    """CER that does not punish a different CONVENTION for the same content.

    A digit run is genuinely ambiguous in speech: "21" is ยี่สิบเอ็ด when said as a quantity and
    สองหนึ่ง when read out as digits, and the audio decides which. Rather than guess, both readings
    are tried and the better score wins -- lenient in the safe direction, because a false REJECT
    costs a good training clip while a false accept is caught by the rest of the panel.

    Variants are capped: past `max_variants` combinations every run is read digit-wise, which is
    the reading Typhoon uses for phone numbers and postcodes (F2's decisive row).
    """
    runs = _DIGIT_RUN.findall(hyp)
    if not runs:
        return cer(ref, hyp)
    if 2 ** len(runs) > max_variants:
        return cer(ref, _DIGIT_RUN.sub(lambda m: thai_digitwise(m.group()), hyp))

    variants = [hyp]
    for run in dict.fromkeys(runs):
        readings = {thai_digitwise(run)}
        if len(run) <= 7:                       # beyond that a cardinal reading is not plausible
            readings.add(thai_cardinal(int(run)))
        variants = [v.replace(run, r) for v in variants for r in readings]
    return min(cer(ref, v) for v in variants)


def thai_core(s: str) -> str:
    """Thai characters only. Drops Latin and digits, which carry the script convention."""
    return "".join(c for c in for_scoring(s) if "฀" <= c <= "๿")


def mad_bounds(xs: list[float], k: float = 4.0) -> tuple[float, float]:
    """Robust outlier bounds: median +/- k * (1.4826 * median-absolute-deviation).

    Median and MAD rather than mean and standard deviation, because the failures we are hunting are
    exactly the extreme values -- they would inflate a standard deviation and hide themselves.
    """
    if not xs:
        return (0.0, math.inf)
    a = np.asarray(xs, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if mad <= 0:                       # a degenerate corpus; fall back to percentiles
        return (float(np.percentile(a, 1)), float(np.percentile(a, 99)))
    return (med - k * mad, med + k * mad)


# --------------------------------------------------------------------------------- cheap detectors

def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.asarray(y, dtype=np.float32)


def is_silent(y: np.ndarray, floor: float = 1e-3) -> bool:
    return y.size == 0 or float(np.max(np.abs(y))) < floor


def longest_internal_silence_s(y: np.ndarray, floor: float = 5e-3, win: int = 400) -> float:
    """Longest run of near-silence that is not at the very start or end.

    An autoregressive model that loses its place often stalls mid-clip. Leading and trailing
    silence are normal and are excluded, so this only fires on a gap inside the speech.
    """
    if y.size == 0:
        return 0.0
    frames = y[: (y.size // win) * win].reshape(-1, win)
    loud = np.abs(frames).max(axis=1) >= floor
    if not loud.any():
        return y.size / SR
    first, last = int(np.argmax(loud)), int(len(loud) - 1 - np.argmax(loud[::-1]))
    run = best = 0
    for v in loud[first:last + 1]:
        run = 0 if v else run + 1
        best = max(best, run)
    return best * win / SR


def repetition_score(y: np.ndarray) -> float:
    """Highest cosine similarity between two NON-ADJACENT half-second windows, 0..1.

    A looping generation repeats a stretch of audio almost exactly, so two windows far apart in
    time end up nearly identical. Adjacent windows are excluded because a held vowel is not a loop.
    Returns 0.0 for clips too short to have two non-adjacent windows.
    """
    import librosa
    if y.size < SR:
        return 0.0
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40, hop_length=256)
    mel = np.log(mel + 1e-8)
    per_win = max(1, int(0.5 * SR / 256))
    n = mel.shape[1] // per_win
    if n < 3:
        return 0.0
    w = mel[:, : n * per_win].reshape(mel.shape[0], n, per_win).mean(axis=2).T   # (n, 40)
    # 🔴 MEAN-CENTRE FIRST, AND THIS IS THE WHOLE DETECTOR. Measured 2026-09-17: without it, REAL
    # Common Voice speech scores median 0.9983 and synthetic 0.9973 -- real speech looks MORE
    # repetitive than a TTS clip, and a threshold of 0.985 rejected 12 of 12 generated clips and
    # would have rejected almost every real one too. The cause is that raw log-mel windows are
    # dominated by the clip's overall spectral tilt, which barely moves inside one recording, so
    # cosine similarity between any two speech windows sits near 1. It was measuring "is this
    # audio", not "is this a loop".
    # Subtracting the clip's own mean window leaves only each window's DEVIATION from that clip's
    # average, which is what actually repeats when a generation loops.
    w = w - w.mean(axis=0, keepdims=True)
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    # 🔴 A LAG SEARCH, NOT MAX PAIRWISE SIMILARITY -- the second thing this detector got wrong.
    # Max-pairwise asks "do any two windows resemble each other", and in real speech two windows
    # always do (same speaker, recurring phonemes, pauses): measured 2026-09-17, real Common Voice
    # reached p90 0.997 against a true loop's 1.000, which cannot be thresholded.
    # A LOOP is different in kind: window i resembles window i+L for MANY CONSECUTIVE i at one
    # fixed period L. So score each lag by its MEAN similarity down the whole diagonal and take the
    # best lag. One coincidentally similar pair can no longer carry the score.
    n_w = w.shape[0]
    best = 0.0
    for lag in range(2, max(3, n_w // 2) + 1):
        if n_w - lag < 2:
            break
        best = max(best, float(np.mean(np.sum(w[:-lag] * w[lag:], axis=1))))
    return best


def alien_chars(text: str) -> str:
    """Characters a Thai transcript should not contain. Non-empty means the model left Thai."""
    return "".join(sorted({c for c in text if not ALLOWED.match(c)}))


def tail_ok(ref: str, hyp: str, n: int = 12, max_cer: float = 0.45) -> bool:
    """Did the END survive? The check CER cannot make -- an average hides a deletion at one edge.

    A clip can score a respectable overall CER while having dropped exactly its final word, and for
    a rare-entity bank the entity is often the last thing said.

    🔴 THIS COMPARES THE TAIL, IT DOES NOT MATCH IT. Until 2026-09-17 it required the reference's
    last characters to appear VERBATIM in the hypothesis, and that produced false rejects on
    ordinary Thai spelling variation -- measured on the first generated batch:

        label  ...ใช่ไหมครับ     biodatlab  ...ใช่มั้ยครับ      (ไหม / มั้ย, same particle)
        label  ...เท่าไรคะ       biodatlab  ...เท่าไหร่คะ        (same word, colloquial spelling)

    Nothing was dropped in either; both were rejected. That is the FOURTH Thai text convention to
    manufacture a wrong answer here, after SARA AM (F1), digits (F2) and script.

    A dropped tail and a respelled tail are different in SIZE, not in kind, so the test is a CER
    over the last `n` characters of each side. A real deletion sends that toward 1.0; a spelling
    variant leaves it small. Verified on the same batch to still catch pathumma ending
    `...ใช่ไหม` where the label ends `...ใช่ไหมครับ` -- a genuinely dropped ครับ.
    """
    r, h = for_scoring(ref), for_scoring(hyp)
    if not r:
        return True
    if not h:
        return False
    return cer_tolerant(r[-n:], h[-n:]) <= max_cer


# ---------------------------------------------------------------------------- the verifier panel
#
# HIS CALL: "just download the best thai asr from each brand include typhoon to verified the vox
# voice." So the panel is the best Thai ASR each org ships, scored on THIS repo's 682 real
# utterances rather than on their model cards.
#
# 🔴 FOUR BRANDS IS NOT FOUR OPINIONS. Three of the four voting members are fine-tunes of the SAME
# OpenAI whisper-large-v3 checkpoint. They will make correlated mistakes, and a unanimous verdict
# from three siblings is weaker than it looks. Qwen3-ASR is the only structurally independent
# voter (different base, LLM-style decoder), and typhoon-asr-realtime is the only other
# architecture family (FastConformer-Transducer) -- which is exactly why it is on the panel but
# not on the jury. Read the per-model columns, not just the vote.
#
# recall/CER below: measured here, `results/` + RESEARCH-PROGRESS.md, 682 real utts, align dp.
PANEL: dict[str, dict] = {
    "biodatlab": {
        "id": "biodatlab/whisper-th-large-v3-combined", "backend": "hf", "votes": True,
        "note": "best entity recall in the field: 0.435 (CER 0.1068). whisper-large-v3 fine-tune",
    },
    "pathumma": {
        "id": "nectec/Pathumma-whisper-th-large-v3", "backend": "hf", "votes": True,
        "note": "NECTEC. best overall CER in the field: 0.0814 (recall 0.351). whisper-large-v3 fine-tune",
    },
    "qwen": {
        "id": "Qwen/Qwen3-ASR-1.7B-hf", "backend": "qwen", "votes": True,
        "note": "Alibaba. recall 0.460 -- the highest measured here -- CER 0.0860. "
                "THE ONLY STRUCTURALLY INDEPENDENT VOTER: not a whisper fine-tune",
    },
    "typhoon": {
        "id": "typhoon-ai/typhoon-whisper-large-v3", "backend": "hf", "votes": True,
        "note": "SCB10X's best Thai accuracy model: CER 0.0924, recall 0.408. whisper-large-v3 fine-tune. "
                "⚠️ same org as the model under test, so it may share a corpus and its blind spots",
    },
    # ON THE PANEL, OFF THE JURY, AND THAT IS THE WHOLE POINT.
    # This IS the model being trained. Letting it vote would keep only the clips it already
    # understands and throw away the training signal (F5's shape). Recording its opinion instead
    # turns the circularity into the most useful column in the output: clips the panel accepted and
    # THIS model could not hear are the ones with something left to teach it.
    "target": {
        "id": "typhoon-ai/typhoon-asr-realtime", "backend": "nemo", "votes": False,
        "note": "THE MODEL UNDER TEST. CER 0.1125, recall 0.328. Recorded, never counted "
                "(--let-target-vote overrides, for a deliberate circularity experiment)",
    },
}
DEFAULT_PANEL = ["biodatlab", "pathumma", "qwen", "typhoon", "target"]


def resolve_panel(spec: str) -> list[str]:
    if spec in ("none", ""):
        return []
    if spec in ("panel", "all"):
        return list(DEFAULT_PANEL)
    out = []
    for name in (x.strip() for x in spec.split(",") if x.strip()):
        if name in PANEL:
            out.append(name)
        else:                                   # a raw HF id
            PANEL[name] = {"id": name, "backend": "hf", "votes": True, "note": "ad hoc"}
            out.append(name)
    return out


def load_verifier(key: str):
    """Return (transcribe(np.float32 @16k) -> str, close()). One model in memory at a time."""
    spec = PANEL[key]
    import torch
    if spec["backend"] == "nemo":
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(
            spec["id"], map_location="cuda" if torch.cuda.is_available() else "cpu")
        m.eval()

        def transcribe(y):
            out = m.transcribe([y], batch_size=1, verbose=False)
            h = out[0] if out else ""
            return (getattr(h, "text", None) or (h if isinstance(h, str) else "")).strip()

        def close():
            # `nonlocal`, not `del`: the closure holds the reference, so deleting the local name
            # both fails (UnboundLocalError) and would not free the VRAM anyway.
            nonlocal m
            m = None
            torch.cuda.empty_cache()
        return transcribe, close

    if spec["backend"] == "qwen":
        # Qwen3-ASR is NOT a Whisper and the generic ASR pipeline cannot drive it: it expects a
        # chat-templated transcription request, and feeding it raw features raises a shape error
        # (found by the smoke test -- "expanded size of the tensor (128) must match (3000)").
        # `apply_transcription_request` is the documented entry point and it takes a numpy array.
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        proc = AutoProcessor.from_pretrained(spec["id"])
        # No `device_map="auto"`: that needs `accelerate`, which this venv does not have, and
        # adding a package to the ASR venv risks the torch downgrade the README warns about.
        # A single 1.7B model fits on the card without sharding, so .to() is enough.
        qmodel = Qwen3ASRForConditionalGeneration.from_pretrained(
            spec["id"], dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        qmodel = qmodel.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        def transcribe(y):
            inputs = proc.apply_transcription_request(audio=y, language="Thai").to(
                qmodel.device, qmodel.dtype)
            out = qmodel.generate(**inputs, max_new_tokens=256)
            gen = out[:, inputs["input_ids"].shape[1]:]
            # raw output carries a language tag and an <asr_text> marker; ask for the text only
            return (proc.decode(gen, return_format="transcription_only")[0] or "").strip()

        def close():
            nonlocal qmodel, proc
            qmodel = proc = None
            torch.cuda.empty_cache()
        return transcribe, close

    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("automatic-speech-recognition", model=spec["id"], device=dev,
                    torch_dtype=torch.float16 if dev == 0 else torch.float32)

    # `language`/`task` are Whisper's forced-decoder tokens and nothing else accepts them --
    # Qwen3-ASR raises ValueError("model_kwargs are not used by the model"). Found by the smoke
    # test, which is the only reason this panel is not one silently unusable juror smaller.
    is_whisper = getattr(pipe.model.config, "model_type", "") == "whisper"
    gen_kwargs = {"language": "th", "task": "transcribe"} if is_whisper else {}

    def transcribe(y):
        out = pipe({"raw": y, "sampling_rate": SR}, generate_kwargs=gen_kwargs)
        return (out.get("text") or "").strip()

    def close():
        nonlocal pipe
        pipe = None
        torch.cuda.empty_cache()
    return transcribe, close


def to_pcm16_and_back(y: np.ndarray) -> np.ndarray:
    """Round-trip through int16, the way every clip reaches :8007 over the wire.

    🔴 Not cosmetic. RESEARCH-PROGRESS.md records that this round-trip changes 23 of 682
    transcripts -- 3.4% -- while leaving aggregate CER flat, so it is invisible in a metrics table
    and decisive for a per-clip gate like this one.
    """
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32) / 32767.0


# ---------------------------------------------------------------------------------------- self-test

def selftest() -> int:
    """Run each detector on a case built to trigger it. No model, no data, ~1 second.

    This exists because a filter that silently never fires is worse than no filter: it would pass
    a corrupt corpus through with a clean report.
    """
    ok = True

    def check(name: str, got, want) -> None:
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:34s} got={got!r} want={want!r}")

    t = np.arange(SR * 2) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    check("silent: detects digital silence", is_silent(np.zeros(SR, np.float32)), True)
    check("silent: passes real audio", is_silent(tone), False)

    gap = np.concatenate([tone[:SR // 2], np.zeros(SR, np.float32), tone[:SR // 2]])
    check("gap: finds a 1 s internal stall", round(longest_internal_silence_s(gap)) == 1, True)
    check("gap: none in continuous audio", longest_internal_silence_s(tone) < 0.1, True)

    # 🔴 THE HOLE THAT LET A BROKEN DETECTOR SHIP. Until 2026-09-17 this proved only that the
    # detector FIRES on a loop; it never proved it stays QUIET on ordinary speech. The first real
    # run rejected 12 of 12 generated clips, and measurement showed real Common Voice scored HIGHER
    # than the synthetic. A detector that always fires is exactly as useless as one that never does,
    # and only one of those failures was being tested for. Both cases are pinned now.
    #
    # A varying signal, because a CONSTANT tone is the wrong probe: mean-centring correctly zeroes
    # it (it never deviates from its own average), so a tiled pure tone now scores ~0.008 and the
    # old assertion would fail for the right reason.
    seg = np.concatenate([(0.3 * np.sin(2 * np.pi * f * np.arange(SR // 4) / SR)).astype(np.float32)
                          for f in (180, 300, 240, 420)])
    check("repetition: catches a tiled loop", repetition_score(np.tile(seg, 4)) > 0.9, True)
    check("repetition: QUIET on varied audio", repetition_score(
        np.concatenate([(0.3 * np.sin(2 * np.pi * f * np.arange(SR // 4) / SR)).astype(np.float32)
                        for f in (180, 300, 240, 420, 150, 500, 210, 660, 270, 330)])) < 0.8, True)

    check("alien: flags CJK", alien_chars("สวัสดี你好") != "", True)
    check("alien: allows Thai + Latin", alien_chars("ไปที่ B2 ครับ"), "")

    check("tail: intact tail passes", tail_ok("ไปตลาดคลองเตย", "ไปตลาดคลองเตย"), True)
    check("tail: dropped tail fails", tail_ok("ไปตลาดคลองเตยตอนเช้านะครับ", "ไปตลาดคลองเตย"), False)
    # 🔴 BOTH DIRECTIONS, from the first generated batch. A tail check that cannot tell a Thai
    # spelling variant from a deletion rejects good audio -- the fourth convention trap today.
    check("tail: spelling variant passes",
          tail_ok("ยังมีประกันศูนย์อยู่ใช่ไหมครับ", "ยังมีประกันศูนย์อยู่ใช่มั้ยครับ"), True)
    check("tail: variant เท่าไร/เท่าไหร่ passes",
          tail_ok("คิดค่าบริการเท่าไรคะ", "คิดค่าบริการเท่าไหร่คะ"), True)
    check("tail: a real dropped ครับ still fails",
          tail_ok("ยังมีประกันศูนย์อยู่ใช่ไหมครับ", "แปดยังมีประกันศูนย์อยู่ใช่ไหม"), False)

    check("cer: identical is 0", cer("ข้าว", "ข้าว"), 0.0)
    check("cer: tone error is caught", cer("ข้าว", "ข่าว") > 0, True)
    check("cer: SARA AM fold, both sides", cer("บำรุง", "บํารุง"), 0.0)

    lo, hi = mad_bounds([10.0] * 20 + [99.0])
    check("mad: outlier lands outside", 99.0 > hi, True)

    # Thai numerals have two irregulars and a digit-by-digit reading, and the verifier writes
    # digits where the label writes words. Measured on the panel's first two real clips.
    check("thai num: 21 is yi-sip-et", thai_cardinal(21), "ยี่สิบเอ็ด")
    check("thai num: 11 is sip-et", thai_cardinal(11), "สิบเอ็ด")
    check("thai num: 20 is yi-sip", thai_cardinal(20), "ยี่สิบ")
    check("tolerant: digits == words", cer_tolerant("จองอีกสองคืน", "จองอีก 2 คืน"), 0.0)
    check("tolerant: phone read digit-wise", cer_tolerant("เบอร์ศูนย์แปดสาม", "เบอร์ 083"), 0.0)
    check("tolerant: irregular cardinal", cer_tolerant("ยี่สิบเอ็ดบาท", "21 บาท"), 0.0)
    # 🔴 THE ONE THAT KEEPS THE REST HONEST. Leniency must not reach a wrong entity.
    check("tolerant: WRONG stays wrong",
          cer_tolerant("ไปตลาดคลองเตย", "ไปตลาดบางกะปิ") > 0.4, True)
    check("thai_core: drops Latin and digits", thai_core("ไปที่ B2 ชั้น 3"), "ไปที่ชั้น")

    # --- the entity gate. Every case below is a real row from the first 12 clips. -----------
    def ent_cer(e, h):
        return round(infix_cer(for_scoring(e), for_scoring(h)), 2)

    check("entity: clean readback is 0",
          ent_cer("ซาร่า", "ยาน้ำลดไข้เด็กซาร่ารสส้มขวดละหกห้าบาท"), 0.0)
    # 🔴 THE CASE THAT JUSTIFIES THE WHOLE GATE. The sentence passed --cer-max 0.10; the name
    # inside it did not survive. An 85-character utterance hides a 10-character wreck.
    check("entity: the name is wrecked, the sentence is not",
          ent_cer("พลับพลาไชย", "ที่ตำบลพัฒนาชัยอำเภอ") >= 0.4, True)
    # 🔴 AND THE CASE THAT SETS THE THRESHOLD ABOVE ZERO. One tone mark. An exact-substring test
    # called four of twelve clips unsayable; every one of them was this.
    check("entity: one tone mark is a near miss, not a failure",
          ent_cer("อุไดเจริญ", "ตำบลอุได้เจริญรหัสไปรษณีย์") <= 0.15, True)
    check("entity: a missing name scores 1.0",
          ent_cer("เมืองปอน", "") , 1.0)

    print("\n  SELFTEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the detectors on built cases, then exit")
    ap.add_argument("--manifest", help="jsonl from colab/gen_colab.py: clip, gen_id, text, dur_s, ...")
    ap.add_argument("--clips", help="directory holding the wavs named in the manifest")
    ap.add_argument("--asr", default="none",
                    help="none (cheap checks only) | panel (every brand, the default jury) | "
                         "a comma list of " + ",".join(PANEL) + " or raw HF ids")
    ap.add_argument("--vote", default="majority", choices=("majority", "unanimous", "any"),
                    help="how the jury combines. majority is the default; unanimous rejects more "
                         "and is riskier because three of the four jurors are whisper siblings")
    ap.add_argument("--let-target-vote", action="store_true",
                    help="let typhoon-asr-realtime -- THE MODEL BEING TRAINED -- reject clips. "
                         "Off by default: it would keep only what it already understands and "
                         "discard the training signal. On, this becomes a circularity experiment")
    ap.add_argument("--prefetch", action="store_true",
                    help="download every panel model and exit. ~7 GB; check disk first")
    ap.add_argument("--cer-max", type=float, default=0.10,
                    help="readback CER above this is a reject. 0.10 is the endpointer line's band; "
                         "CALIBRATE IT BY EAR before trusting it")
    ap.add_argument("--rep-max", type=float, default=0.80,
                    help="mean lag-similarity above this is a loop. CALIBRATED 2026-09-17 on "
                         "this box: real Common Voice maxes at 0.489 and VoxCPM synthetic at "
                         "0.620, while a real clip tiled 3x scores 0.999. 0.80 sits in that gap")
    ap.add_argument("--entity-max", type=float, default=0.25,
                    help="for a phase-2 row, how far a juror's readback of the RARE ENTITY may sit "
                         "from the label, as a CER over the entity alone. Whole-utterance CER "
                         "cannot see this: an 8-character name inside an 85-character sentence "
                         "can be destroyed while the sentence still passes --cer-max. Measured "
                         "2026-09-17 on the first 12 clips -- see this flag's note in HISTORY")
    ap.add_argument("--entity-votes", type=int, default=2,
                    help="how many jurors must clear --entity-max for a phase-2 clip to survive. "
                         "0 disables the entity gate entirely")
    ap.add_argument("--gap-max", type=float, default=1.5, help="internal silence (s) above this is a stall")
    ap.add_argument("--mad-k", type=float, default=4.0, help="duration outlier width, in MADs")
    ap.add_argument("--calibrate-sample", type=int, default=0,
                    help="also write N randomly chosen clips to a listen/ dir, half kept and half "
                         "rejected, so the thresholds can be checked against ears")
    ap.add_argument("--tag", default="synth", help="prefix for the files written to results/")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (a smoke run)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.prefetch:
        # Pull every panel model now so a long verification run does not stall on a download
        # halfway through, and so the disk cost is paid visibly rather than by surprise.
        from huggingface_hub import snapshot_download
        for key in (resolve_panel(args.asr) or DEFAULT_PANEL):
            spec = PANEL[key]
            print(f"[{key}] {spec['id']}")
            print(f"    {spec['note']}")
            snapshot_download(spec["id"])
        print("panel on disk")
        return 0

    if not args.manifest or not args.clips:
        ap.error("--manifest and --clips are required unless --selftest or --prefetch")

    clips_dir = Path(args.clips)
    rows = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("empty manifest", file=sys.stderr)
        return 1
    print(f"{len(rows)} clips from {args.manifest}")

    # --- pass 1: cheap, and it is what sets the duration bounds -------------------------------
    feats: list[dict] = []
    for i, r in enumerate(rows, 1):
        path = clips_dir / r["clip"]
        if not path.exists():
            feats.append({**r, "fail": ["missing_file"]})
            continue
        y = load_wav(path)
        text = r.get("label_text") or r["text"]
        n_chars = max(1, len(for_scoring(text)))
        dur = y.size / SR
        feats.append({
            **r, "dur_measured": dur, "n_chars": n_chars,
            "chars_per_s": n_chars / dur if dur > 0 else 0.0,
            "silent": is_silent(y),
            "gap_s": longest_internal_silence_s(y),
            "rep": repetition_score(y),
            "fail": [],
        })
        if i % 250 == 0:
            print(f"  cheap checks {i}/{len(rows)}", flush=True)

    rates = [f["chars_per_s"] for f in feats if f.get("chars_per_s")]
    lo, hi = mad_bounds(rates, args.mad_k)
    print(f"\nchars/second over this corpus: median {np.median(rates):.2f}  "
          f"accepted band [{lo:.2f}, {hi:.2f}]  (median +/- {args.mad_k} MAD, derived here -- "
          f"not a constant)")

    for f in feats:
        if "missing_file" in f["fail"]:
            continue
        if f["silent"]:
            f["fail"].append("silent")
        if f["chars_per_s"] < lo:
            f["fail"].append("too_long_for_text")     # few chars, lots of audio -> ran on
        if f["chars_per_s"] > hi:
            f["fail"].append("too_short_for_text")    # many chars, little audio -> skipped
        if f["rep"] > args.rep_max:
            f["fail"].append("repetition")
        if f["gap_s"] > args.gap_max:
            f["fail"].append("internal_stall")

    # --- pass 2: the panel, one model at a time -----------------------------------------------
    panel = resolve_panel(args.asr)
    jury: list[str] = []
    if panel:
        todo = [f for f in feats if not f["fail"]]
        print()
        print(f"{len(todo)} clips survived the cheap checks; panel of {len(panel)}")
        for f in todo:
            f["hyps"], f["cers"], f["accept"] = {}, {}, {}
            f["cers_strict"], f["script_variant"] = {}, {}

        for key in panel:
            spec = PANEL[key]
            voting = spec["votes"] and (key != "target" or args.let_target_vote)
            if voting:
                jury.append(key)
            print()
            print(f"[{key}] {spec['id']}   {'VOTES' if voting else 'recorded, no vote'}")
            print(f"      {spec['note']}")
            transcribe, close = load_verifier(key)
            try:
                for i, f in enumerate(todo, 1):
                    y = to_pcm16_and_back(load_wav(clips_dir / f["clip"]))
                    ref = f.get("label_text") or f["text"]
                    hyp = transcribe(y)
                    c_strict, c_gate = cer(ref, hyp), cer_tolerant(ref, hyp)
                    f["hyps"][key] = hyp
                    f["cers"][key] = round(c_gate, 4)
                    f["cers_strict"][key] = round(c_strict, 4)
                    # ⚠️ SCRIPT MISMATCH IS NOT SCORED, IT IS ROUTED. fix_label_convention.py makes
                    # every label Thai-only, so Latin in a hypothesis means the verifier wrote a
                    # name in Latin where the label has it in Thai. That is the same convention
                    # problem as digits, but it needs transliteration to score -- a separate arm.
                    # Until it exists, these go to review instead of being rejected as wrong words.
                    f["script_variant"][key] = bool(re.search(r"[A-Za-z]", hyp)
                                                    and not re.search(r"[A-Za-z]", ref))
                    f["accept"][key] = bool(c_gate <= args.cer_max and not alien_chars(hyp)
                                            and tail_ok(ref, hyp))
                    # The one word the phase-2 bank exists to teach, scored on its own.
                    ent = f.get("entity")
                    if ent:
                        f.setdefault("entity_cer", {})[key] = round(
                            infix_cer(for_scoring(ent), for_scoring(hyp)), 4)
                    if i % 200 == 0:
                        print(f"      {i}/{len(todo)}", flush=True)
            finally:
                close()

        if not jury:
            print()
            print("[!] no voting member in the panel -- readback can reject nothing",
                  file=sys.stderr)
        need = {"unanimous": len(jury), "any": 1}.get(args.vote, (len(jury) // 2) + 1)
        print()
        print(f"vote = {args.vote}: a clip needs {need} of {len(jury)} jurors")
        for f in todo:
            n_yes = sum(1 for k in jury if f["accept"].get(k))
            f["n_accept"] = n_yes
            if jury and n_yes < need:
                f["fail"].append("panel_rejected")

        # --- the entity gate, and why the utterance gate above cannot replace it ----------------
        #
        # 🔴 MEASURED 2026-09-17 ON THE FIRST 12 CLIPS. All 12 passed the panel at --cer-max 0.10.
        # Scored on the rare entity ALONE, two of them were wrecked: `พลับพลาไชย` came back as
        # `พัฒนาชัย` / `พรรคพราชัย` (entity CER 0.40-0.60 for three of four jurors) inside a
        # sentence that still scored under 0.10, because the name is 10 characters of 85.
        #
        # For `phase1_known` that is fine -- the clip teaches general Thai either way. For
        # `phase2_rare` the entity IS the clip: a sentence built around `พลับพลาไชย` that does not
        # contain a sayable `พลับพลาไชย` teaches the model a wrong spelling for a sound. That is
        # worse than not training on it, and `H37`'s warning -- a rare-word ceiling no
        # self-distillation overcomes -- is exactly this failure surviving a verifier.
        #
        # ⚠️ THE THRESHOLD IS NOT AN EXACT MATCH, DELIBERATELY. An exact-substring test on those
        # same 12 clips called FOUR of them unsayable; graded, all four were one tone mark or one
        # letter out (`อุไดเจริญ` heard as `อุได้เจริญ`, 0.11). Exact matching on Thai rare names
        # reproduces the very error `F17` is about, and it nearly went into the record as a finding.
        if jury and args.entity_votes > 0:
            n_ent_gated = 0
            for f in todo:
                if not f.get("entity") or f["fail"]:
                    continue
                n_ok = sum(1 for k in jury
                           if f.get("entity_cer", {}).get(k, 1.0) <= args.entity_max)
                f["n_entity_ok"] = n_ok
                if n_ok < args.entity_votes:
                    f["fail"].append("entity_unheard")
                    n_ent_gated += 1
            print(f"entity gate: <= {args.entity_max} entity-CER from >= {args.entity_votes} "
                  f"jurors -- rejected {n_ent_gated} clip(s) the utterance gate had passed")

    # --- report --------------------------------------------------------------------------------
    keep = [f for f in feats if not f["fail"]]
    reject = [f for f in feats if f["fail"]]
    reasons = Counter(r for f in reject for r in f["fail"])

    print(f"\n{'=' * 62}\nkeep   {len(keep):5d} / {len(feats)}   ({len(keep) / len(feats):.1%})")
    print(f"reject {len(reject):5d}\n")
    for reason, n in reasons.most_common():
        print(f"  {reason:22s} {n:5d}  {n / len(feats):6.1%}")

    per_model: dict[str, dict] = {}
    if panel:
        judged = [f for f in feats if f.get("cers")]
        print()
        print("per model, over the clips that reached the panel:")
        print(f"  {'model':11s} {'votes':6s} {'accepts':>8s}  {'CER med':>8s} {'p90':>7s}")
        for key in panel:
            cs = [f["cers"][key] for f in judged if key in f["cers"]]
            if not cs:
                continue
            acc = sum(1 for f in judged if f["accept"].get(key))
            per_model[key] = {"accept_rate": acc / len(cs), "cer_median": float(np.median(cs)),
                              "cer_p90": float(np.percentile(cs, 90)), "voted": key in jury}
            print(f"  {key:11s} {'yes' if key in jury else 'no':6s} "
                  f"{acc / len(cs):7.1%}  {np.median(cs):8.3f} {np.percentile(cs, 90):7.3f}")

        # * THE COLUMN THIS PANEL EXISTS FOR. The jury heard it; the model being trained did not.
        # Those clips are not rejects -- they are the ones with something left to teach it.
        need2 = max(1, len(jury) // 2 + 1)
        teach = [f for f in judged
                 if "target" in f.get("accept", {}) and not f["accept"]["target"]
                 and f.get("n_accept", 0) >= need2]
        print()
        print(f"* panel accepted, MODEL UNDER TEST could not hear: {len(teach)} clips "
              f"({len(teach) / max(1, len(judged)):.1%})")
        print("  these are the highest-value training clips, not failures "
              "-- written to *_verify_teach.jsonl")
        (OUT / f"{args.tag}_verify_teach.jsonl").write_text(
            "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in teach), encoding="utf-8")

        # three of four jurors are whisper-large-v3 fine-tunes, so measure how correlated they are
        if len(jury) > 1:
            print()
            print("pairwise juror agreement (whisper siblings will look independent and are not):")
            for i, k1 in enumerate(jury):
                for k2 in jury[i + 1:]:
                    both = [f for f in judged if k1 in f["accept"] and k2 in f["accept"]]
                    if both:
                        same = sum(1 for f in both if f["accept"][k1] == f["accept"][k2])
                        print(f"  {k1:11s} vs {k2:11s} {same / len(both):6.1%}")

    # rejection by text, so a text that never generates cleanly is visible as a HARD CASE rather
    # than quietly vanishing from the corpus.
    by_text: dict[str, list[bool]] = {}
    for f in feats:
        by_text.setdefault(str(f.get("text_id", f["gen_id"])), []).append(not f["fail"])
    hard = [t for t, oks in by_text.items() if not any(oks)]
    print(f"\ntexts with NO usable clip: {len(hard)} of {len(by_text)}"
          + ("   <-- these are hard cases, not losses" if hard else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}_verify_keep.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in keep), encoding="utf-8")
    (OUT / f"{args.tag}_verify_reject.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in reject), encoding="utf-8")
    report = {
        "n": len(feats), "keep": len(keep), "reject": len(reject),
        "keep_rate": len(keep) / len(feats),
        "reasons": dict(reasons),
        "chars_per_s": {"median": float(np.median(rates)), "band": [lo, hi], "mad_k": args.mad_k},
        "thresholds": {"cer_max": args.cer_max, "rep_max": args.rep_max, "gap_max": args.gap_max},
        "asr": args.asr, "hard_case_texts": hard,
        "note": "CER here is a gate, not an arm number -- adapter/_score.py pins the backend, this does not.",
    }
    (OUT / f"{args.tag}_verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / (args.tag + '_verify_report.json')}")

    if args.calibrate_sample:
        # Half kept, half rejected, so the ear sees both kinds. The point is to find out whether
        # the thresholds agree with him -- not to admire the ones that passed.
        listen = OUT / f"{args.tag}_listen"
        listen.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20260916)
        half = max(1, args.calibrate_sample // 2)
        picks = rng.sample(keep, min(half, len(keep))) + rng.sample(reject, min(half, len(reject)))
        import shutil
        for f in picks:
            verdict = "KEEP" if not f["fail"] else "REJECT-" + "+".join(f["fail"])
            src = clips_dir / f["clip"]
            if src.exists():
                shutil.copy(src, listen / f"{verdict}__{f['clip']}")
        (listen / "_texts.txt").write_text(
            "".join(f"{f['clip']}\t{f.get('label_text') or f['text']}\t"
                    + " | ".join(f"{k}={v}" for k, v in (f.get("hyps") or {}).items()) + "\n"
                    for f in picks), encoding="utf-8")
        print(f"LISTEN to {len(picks)} clips in {listen} and move the CER cut to where you agree.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
