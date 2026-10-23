#!/usr/bin/env python3
"""YODAS2-th UPLOADER CAPTIONS -> NECTEC training clips, confirmed against the audio by two ears.

    PYTHONUTF8=1 python adapter/yodas_prep.py --stage prep      # convention + leak guard + NECTEC pass -> hard list
    E:/fw-spike/.venv/Scripts/python.exe adapter/fw_batch.py --model <Pathumma-L> \\
        --manifest E:/Claude/datasets/yodas2_th/manifest_yodas_hard.jsonl --out E:/Claude/datasets/yodas2_th/patL_yodas_hard.jsonl
    PYTHONUTF8=1 python adapter/yodas_prep.py --stage confirm   # -> manifest_yodas_conf.jsonl

WHY (2026-10-11, loop iteration 11). GigaSpeech2-train MACHINE labels failed twice (`F48`, iteration 6): on the
clips NECTEC misses, the label is mostly what is wrong. YODAS2's 47,956 Thai segments on disk (~50 h, CC-BY-3.0)
carry the uploaders' own captions -- human-written, and unlike GS2 they keep foreign names in LATIN, the
benchmark's convention ("WatchOS", "Watch Face"). Subtitles are not always verbatim, so the same two-ear
confirmation is applied:

  prep    : caption -> fleurs_prep.convention (lowercase, digits -> Thai number words, SARA AM composed, Thai-Thai
            spaces dropped), allowed chars [Thai a-z space] only, 1-20 s, 20-char LEAK GUARD vs the benchmark.
            NECTEC transcribes all; agreement is measured on THAI characters only (NECTEC rarely writes Latin).
            FRESHNESS (`F28`) is read off the same pass: NECTEC's Thai-char CER vs the captions.
  confirm : easy (NECTEC exact on Thai chars) -> keep a 15% sample; hard -> keep if Pathumma-L's Thai-char CER vs
            the caption <= .03 AND Pathumma-L is closer than NECTEC.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from fleurs_prep import ALLOWED, convention, lev  # noqa: E402
from gs2_convention import BENCH  # noqa: E402

D = Path("E:/Claude/datasets/yodas2_th")
NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"
NOT_THAI = re.compile(r"[^\u0e00-\u0e7f]")


def thai(t: str) -> str:
    return NOT_THAI.sub("", canon(t).replace("\u0e4d\u0e32", "\u0e33"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prep", "confirm"), required=True)
    args = ap.parse_args()
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    conv, nec_path = D / "manifest_yodas_conv.jsonl", D / "nectec_yodas.jsonl"
    if args.stage == "prep":
        grams = set()
        for l in BENCH.open(encoding="utf-8"):
            t = re.sub(r"\s", "", json.loads(l)["text"]).lower()
            grams.update(t[i:i + 20] for i in range(len(t) - 19))
        rows, drop, seen = [], {"leak": 0, "chars": 0, "dur": 0, "dup": 0, "empty": 0}, set()
        for p in ("manifest.jsonl", "manifest_s3_12.jsonl"):
            for l in (D / p).open(encoding="utf-8"):
                r = json.loads(l)
                if r["id"] in seen:
                    drop["dup"] += 1; continue
                seen.add(r["id"])
                raw = re.sub(r"\s", "", r["caption"]).lower()
                if any(raw[i:i + 20] in grams for i in range(len(raw) - 19)):
                    drop["leak"] += 1; continue
                text = convention(r["caption"])
                if not text or not thai(text):
                    drop["empty"] += 1; continue
                if not ALLOWED.match(text):
                    drop["chars"] += 1; continue
                if not 1.0 <= r["dur"] <= 20.0:
                    drop["dur"] += 1; continue
                rows.append({"audio_filepath": r["wav"].replace("\\", "/"), "duration": r["dur"], "text": text,
                             "id": r["id"], "video": r["video"]})
        with conv.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lat = sum(1 for r in rows if re.search("[a-z]", r["text"]))
        print(f"kept {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.1f} h, Latin {lat}; dropped {drop}",
              flush=True)
        done = {}
        if nec_path.exists():
            done = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in nec_path.open(encoding="utf-8")}
        todo = [r for r in rows if r["id"] not in done]
        if todo:
            import nemo.collections.asr as nemo_asr
            import torch
            # cap torch's cache: on Windows a full VRAM spills into system RAM instead of raising OOM
            # (2026-10-11: batch 32 reached 22 GB private and 0.7 GB commit in a minute)
            torch.cuda.set_per_process_memory_fraction(0.85)
            m = nemo_asr.models.ASRModel.restore_from(NEC, map_location="cuda")
            m.eval()
            with nec_path.open("a", encoding="utf-8", newline="\n") as f:
                for i in range(0, len(todo), 256):
                    chunk = todo[i: i + 256]
                    with torch.inference_mode():
                        out = m.transcribe([r["audio_filepath"] for r in chunk], batch_size=16, verbose=False)
                    for r, h in zip(chunk, out):
                        t = (h.text if hasattr(h, "text") else str(h)).strip()
                        done[r["id"]] = t
                        f.write(json.dumps({"id": r["id"], "hyp": t}, ensure_ascii=False) + "\n")
                    if i % 5120 == 0:
                        print(f"  NECTEC {i}/{len(todo)}", flush=True)
        E = C = 0
        hard = []
        for r in rows:
            a, b = thai(r["text"]), thai(done[r["id"]])
            e = lev(a, b); E += e; C += len(a)
            if e > 0:
                hard.append(r)
        with (D / "manifest_yodas_hard.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in hard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ = {"kept": len(rows), "latin_rows": lat, "dropped": drop, "nectec_thai_cer_vs_caption": round(E / C, 4),
                "hard": len(hard), "easy": len(rows) - len(hard)}
        (HERE.parent / "results/yodas_prep.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
        print(json.dumps(summ), flush=True)
        return 0

    rows = [json.loads(l) for l in conv.open(encoding="utf-8")]
    nec = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in nec_path.open(encoding="utf-8")}
    pat = {json.loads(l)["id"]: json.loads(l)["text"] for l in (D / "patL_yodas_hard.jsonl").open(encoding="utf-8")}
    rng = random.Random(2026)
    keep, n_easy, n_hard = [], 0, 0
    for r in rows:
        a = thai(r["text"])
        en = lev(a, thai(nec[r["id"]]))
        if en == 0:
            if rng.random() < 0.15:
                keep.append(r); n_easy += 1
            continue
        p = pat.get(r["id"])
        if p is None:
            continue
        ep = lev(a, thai(p))
        if ep / max(len(a), 1) <= 0.03 and ep < en:
            keep.append(r); n_hard += 1
    with (D / "manifest_yodas_conf.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in keep:
            f.write(json.dumps({k: r[k] for k in ("audio_filepath", "duration", "text", "id")}, ensure_ascii=False) + "\n")
    lat = sum(1 for r in keep if re.search("[a-z]", r["text"]))
    print(f"confirmed hard {n_hard} + easy {n_easy} = {len(keep)} clips, "
          f"{sum(r['duration'] for r in keep) / 3600:.1f} h, Latin {lat}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
