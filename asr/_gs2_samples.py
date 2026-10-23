"""Pull a handful of GigaSpeech2 Thai clips to listen to, without downloading a gigabyte.

WHY THIS EXISTS. GigaSpeech2 is the corpus behind 93.9% of Typhoon ASR's training data
(arXiv 2601.13044 SS2.3, 10,329.5 h of the 11,000 h total, pseudo-labelled by three Thai
Whisper-Large models voting). Before trusting anything downstream of Typhoon it is worth
hearing what it was actually taught on -- and the transcripts here are MACHINE-WRITTEN, so
the interesting question is not just how the audio sounds but whether the text is right.

HOW IT AVOIDS THE 966 MB. `dev.tar.gz` is a gzip stream, so it is not seekable -- but it IS
sequential, and tarfile's "r|gz" mode reads it as a stream. We open the HF resolve URL with
`stream=True` and break out of the member loop after N wavs, so only the first few MB ever
cross the wire. Do NOT change "r|gz" to "r:gz": that seeks, and seeking forces the whole file.

ACCESS. The dataset is gated (auto-approve). Two one-time steps:
  1. Accept the terms at https://huggingface.co/datasets/speechcolab/gigaspeech2
  2. `hf auth login`  (or set HF_TOKEN)
LICENCE. The access agreement is NON-COMMERCIAL research/education only, and SpeechColab
does not hold copyright on the audio (it is scraped YouTube). Listening is fine. Training a
shipped model on it is not. Same shape of problem as the un-consented mint_v4 voice.

USAGE
  E:/voxcpm-spike/.venv/Scripts/python.exe asr/_gs2_samples.py            # 12 dev clips
  E:/voxcpm-spike/.venv/Scripts/python.exe asr/_gs2_samples.py -n 30 --split test
"""

import argparse
import io
import os
import sys
import tarfile
import wave
from pathlib import Path

import requests
from huggingface_hub import get_token

REPO = "speechcolab/gigaspeech2"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/th"
OUT = Path(__file__).resolve().parent.parent / "voice-samples" / "gigaspeech2_th"


def _headers():
    tok = get_token() or os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit(
            "No Hugging Face token.\n"
            "  1. accept the terms: https://huggingface.co/datasets/speechcolab/gigaspeech2\n"
            "  2. hf auth login\n"
            "Then re-run."
        )
    return {"Authorization": f"Bearer {tok}"}


def load_transcripts(split, headers):
    """dev.tsv / test.tsv are ~1.5 MB -- small enough to just take whole. sid <TAB> text."""
    r = requests.get(f"{BASE}/{split}.tsv", headers=headers, timeout=120)
    if r.status_code in (401, 403):
        sys.exit(f"HTTP {r.status_code} on {split}.tsv -- terms not accepted, or token lacks access.")
    r.raise_for_status()
    out = {}
    for line in r.content.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0].strip()] = parts[1].strip()
    return out


def stream_wavs(split, n, headers):
    """Read the tar as a STREAM and stop after n wavs. Only the first few MB is transferred."""
    url = f"{BASE}/{split}.tar.gz"
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        if r.status_code in (401, 403):
            sys.exit(f"HTTP {r.status_code} on {split}.tar.gz -- terms not accepted, or token lacks access.")
        r.raise_for_status()
        got = []
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                got.append((member.name, fh.read()))
                if len(got) >= n:
                    break
        return got


def describe(raw):
    with wave.open(io.BytesIO(raw)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth() * 8, w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12, help="how many clips")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()

    headers = _headers()
    print(f"reading {args.split}.tsv ...")
    texts = load_transcripts(args.split, headers)
    print(f"  {len(texts)} transcripts")

    print(f"streaming {args.split}.tar.gz, stopping after {args.n} wavs ...")
    clips = stream_wavs(args.split, args.n, headers)

    outdir = OUT / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, raw in clips:
        sid = Path(name).stem
        dest = outdir / f"{sid}.wav"
        dest.write_bytes(raw)
        total += len(raw)
        sr, ch, bits, dur = describe(raw)
        print(f"\n  {dest.name}  {sr} Hz  {ch}ch  {bits}-bit  {dur:.2f}s")
        print(f"    {texts.get(sid, '(no transcript row -- sid not in tsv)')}")

    print(f"\n{len(clips)} clips, {total/1e6:.1f} MB -> {outdir}")
    print("Listen, then judge the TEXT as well as the audio: these transcripts were written")
    print("by three Whisper models voting, not by a person.")


if __name__ == "__main__":
    main()
