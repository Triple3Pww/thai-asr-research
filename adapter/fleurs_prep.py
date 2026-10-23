#!/usr/bin/env python3
"""FLEURS-th TRAIN -> NECTEC fine-tuning manifest in the benchmark's convention, plus a freshness check.

    PYTHONUTF8=1 python adapter/fleurs_prep.py                 # manifest + NECTEC/gs2lot on 300 clips

WHY (2026-10-11, loop iteration 5). On the benchmark DEV half the "c" utterances (240 of 339, 60% of the best
model's errors) are FLEURS-th dev/test SENTENCES: their references equal FLEURS's normalised transcripts
(lowercase Latin for foreign names, digits that `eval_canon.canon` verbalises). GigaSpeech2 moved the
conversational "a" half -17% but the "c" half not at all (gs2lot .0359 vs NECTEC .0353) -- GS2 has no Latin,
and these errors are mostly foreign names that the references write in Latin. FLEURS TRAIN is a disjoint set of
1,408 sentences (2,602 readings, ~13 h, CC-BY-4.0) whose labels write such names in Latin (16% of rows).

  * labels = FLEURS normalised column -> `canon`-style digits to Thai number words (the scorer's own
    `thai_cardinal`), SARA AM composed (NECTEC writes it), spaces dropped between Thai characters and kept
    next to Latin (NECTEC's own habit), hyphens -> space; rows with any other character dropped
  * LEAK GUARD: drop every reading whose sentence shares a 20-character run with ANY of the 2,062 benchmark
    references (catches the ~40 FLEURS-train sentences that recur in the benchmark)
  * freshness (`F28`): NECTEC and gs2lot_a75 on 300 random kept clips, plain CER under canon; compare with
    NECTEC's .035 on the benchmark's c-dev. Near-perfect would mean it was trained on FLEURS train.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon, thai_cardinal  # noqa: E402
from gs2_convention import BENCH  # noqa: E402

D = Path("E:/Claude/datasets/fleurs_th")
NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"
ALLOWED = re.compile(r"^[\u0e00-\u0e7fa-z ]+$")
THAI = "\u0e00-\u0e7f"


def convention(t: str) -> str:
    t = t.lower().replace("-", " ").replace("\u0e4d\u0e32", "\u0e33")
    t = re.sub(r"\d+", lambda m: thai_cardinal(int(m.group())) if len(m.group()) <= 9 else m.group(), t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(rf"(?<=[{THAI}]) (?=[{THAI}])", "", t)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    import soundfile as sf
    grams = set()
    for l in BENCH.open(encoding="utf-8"):
        t = re.sub(r"\s", "", json.loads(l)["text"]).lower()
        grams.update(t[i:i + 20] for i in range(len(t) - 19))
    rows, drop = [], {"leak": 0, "chars": 0, "dur": 0, "missing": 0}
    for l in (D / "train.tsv").open(encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        raw = re.sub(r"\s", "", f[2]).lower()
        if any(raw[i:i + 20] in grams for i in range(len(raw) - 19)):
            drop["leak"] += 1; continue
        text = convention(f[3])
        if not ALLOWED.match(text):
            drop["chars"] += 1; continue
        wav = D / "audio/train" / f[1]
        if not wav.exists():
            drop["missing"] += 1; continue
        d = sf.info(str(wav)).duration
        if not 1.0 <= d <= 20.0:
            drop["dur"] += 1; continue
        rows.append({"audio_filepath": str(wav).replace("\\", "/"), "duration": round(d, 2), "text": text,
                     "id": f"fl{f[0]}-{f[1][:-4]}"})
    with (D / "manifest_train_conv.jsonl").open("w", encoding="utf-8", newline="\n") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = sum(1 for r in rows if re.search("[a-z]", r["text"]))
    print(f"kept {len(rows)} readings / {len({r['id'].split('-')[0] for r in rows})} sentences, "
          f"{sum(r['duration'] for r in rows) / 3600:.1f} h, Latin rows {lat}; dropped {drop}", flush=True)

    sample = random.Random(0).sample(rows, 300)
    import nemo.collections.asr as nemo_asr
    import torch
    out = {}
    for name, path in (("nectec", NEC), ("gs2lot_a75", "checkpoints/gs2lot_a75.nemo")):
        m = nemo_asr.models.ASRModel.restore_from(path, map_location="cuda"); m.eval()
        with torch.inference_mode():
            hyp = m.transcribe([r["audio_filepath"] for r in sample], batch_size=16, verbose=False)
        E = C = 0
        for r, h in zip(sample, hyp):
            ref = canon(r["text"]).lower(); hy = canon(h.text if hasattr(h, "text") else str(h)).lower()
            E += lev(ref, hy); C += len(ref)
        out[name] = round(E / C, 4)
        print(name, "CER on 300 FLEURS-train clips", out[name], flush=True)
        del m; torch.cuda.empty_cache()
    (HERE.parent / "results/fleurs_train_fresh.json").write_text(json.dumps(
        {"kept": len(rows), "latin_rows": lat, "dropped": drop, "fresh300": out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
