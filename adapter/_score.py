#!/usr/bin/env python3
"""The benchmark's own scorer, with its alignment backend PINNED.

    python adapter/_score.py --hyps <file> --out <file> [--align dp|kaldialign]

WHY THIS EXISTS -- a real defect, found 2026-09-12.

`wayu-ai/thai-contextasr-bench`'s scorer says of its optional dependency:

    Pure stdlib. `kaldialign` is used if importable (faster), else a deterministic
    O(nm) DP with match > sub > del > ins tie-breaking.                -- metrics.py:15

"faster" implies the two paths agree. They do not. Measured on ONE byte-identical hypothesis
file (682 utts, `d0_rw_greedy_hyps.jsonl`), the same scorer, three repeats each:

    backend          CER       U-CER     B-CER     recall
    DP (no kaldialign)  0.11248   0.10571   0.26178   0.32802
    kaldialign          0.11248   0.10622   0.25050   0.32802
                        ------    +0.0005   -0.0113   ------
                        same                4.3% rel   same

CER and entity recall do not move -- they need no span attribution. B-CER and U-CER DO: they
attribute each edit to "inside a gold-entity span" or "outside" via the alignment, and the two
implementations break ties differently. So the two metrics this whole experiment is gated on
depend on **whether a package happens to be installed in the interpreter that runs the scorer**.

That is not hypothetical here: `E:/asr-spike/.venv` has kaldialign (since 2026-08-17),
`E:/voxcpm-spike/.venv` does not. The arm-1 numbers and the seven-system comparison table in
`E:/Claude/asr-research/RESEARCH-PROGRESS.md` were produced by the VoxCPM interpreter, so they are all on the **DP**
path and mutually consistent -- but a later arm run from the ASR venv would have silently moved
to the other path and shown a 0.011 B-CER "improvement" that is pure tie-breaking.

`--align dp` is therefore the DEFAULT: it is what every number already published in this repo
used, and comparability across arms beats any claim that one alignment is more correct. The
block is done by poisoning `sys.modules`, not by uninstalling anything, so the choice is a flag
rather than a property of whichever venv you happened to use.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

BENCH = Path("E:/asr-bench/thai-contextasr-bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotations", default=str(BENCH / "data/utterances.jsonl"))
    ap.add_argument("--lists", default=str(BENCH / "data/bias_lists.jsonl.gz"))
    ap.add_argument("--align", default="dp", choices=("dp", "kaldialign"),
                    help="dp = the benchmark's pure-Python fallback, and what every number in "
                         "this repo used. kaldialign = its optional fast path. They DISAGREE on "
                         "B-CER/U-CER; see this file's docstring.")
    args = ap.parse_args()

    sys.path.insert(0, str(BENCH / "scorer"))
    if args.align == "dp":
        # An entry of None makes `import kaldialign` raise ImportError, which is exactly the
        # branch metrics.py takes when the package is absent. No uninstall, no venv juggling.
        sys.modules["kaldialign"] = None
    else:
        import kaldialign  # noqa: F401  -- fail loudly here rather than silently falling back

    import metrics                                        # noqa: E402
    backend = "kaldialign" if getattr(metrics, "kaldialign", None) is not None else "dp"
    if backend != args.align:
        print(f"ALIGN BACKEND MISMATCH: asked {args.align}, got {backend}", file=sys.stderr)
        return 2
    print(f"[score] align backend = {backend}", flush=True)

    sys.argv = ["score.py", "--annotations", args.annotations, "--lists", args.lists,
                "--hyps", args.hyps, "--out", args.out]
    runpy.run_path(str(BENCH / "scorer/score.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
