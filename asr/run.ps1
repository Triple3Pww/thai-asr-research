#requires -Version 5.1
# Start the Typhoon Thai ASR server. Runs in the ASR venv -- NOT the VoxCPM one:
# NeMo pulls its own torch, and installing it beside VoxCPM risks a downgrade that breaks
# the live TTS.
#
# !! IT RUNS ON THE GPU BY DEFAULT, and always did. This script and server.py both said "CPU"
# until 2026-09-12: NeMo's `from_pretrained` with no `map_location` lands on cuda:0 whenever torch
# sees a card. The GPU copy holds ~0.95 GB of VRAM. `-Device cpu` is the real CPU, and costs
# RTF 0.027 vs 0.010 (~96 ms more on a 6 s utterance) to give that VRAM back.
#
#   .\run.ps1                                  -> :8007, greedy_batch, GPU (the deployed default)
#   .\run.ps1 -Device cpu                      -> CPU; frees ~0.95 GB of VRAM
#   .\run.ps1 -Strategy malsd_batch -Beam 2    -> batched beam: -8.9% CER, RTF 0.010 -> 0.016
#   .\run.ps1 -NgramLm <path.nemo> -Alpha 0.1  -> + 4-gram shallow fusion
#                                                 WARNING: best CER, but RTF 0.136 -- NOT for the live loop
#   .\run.ps1 -BoostFile <phrases.txt> -BoostAlpha 1.0   -> context-biasing boosting tree
#   .\run.ps1 -Port 9007
#
# Arm results: E:/Claude/asr-research/results/decoding_arms_summary.json. /health reports what is actually live.
#
# MODEL (2026-10-05): the default is now NECTEC's Pathumma FastConformer-RNNT -- CER 0.0676 on the
# 682 real benchmark utterances vs typhoon-asr-realtime 0.0945 and typhoon-whisper-large-v3 0.0735,
# 113 ms/turn at beam 5 (lead over large-v3 not significant, P=0.084; worse on named entities).
# It writes digits and composed SARA AM; Typhoon never does. Record: idea-stage/IDEA_REPORT.md.
#   .\run.ps1                                  -> NECTEC, malsd_batch beam 5 (the new default)
#   .\run.ps1 -Model typhoon                   -> typhoon-asr-realtime, greedy_batch (the old default)
#   .\run.ps1 -Model <hf-name or path.nemo>    -> anything else NeMo can load
param(
    [string]$Model = "nectec",
    [int]$Port = 8007,
    [ValidateSet("cpu", "cuda")] [string]$Device = "cuda",
    [ValidateSet("greedy", "greedy_batch", "beam", "tsd", "alsd", "maes", "malsd_batch", "maes_batch")]
    [string]$Strategy = "greedy_batch",
    [int]$Beam = 2,
    [string]$NgramLm = "",
    [double]$Alpha = 0.0,
    [string]$BoostFile = "",
    [double]$BoostAlpha = 0.0
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "E:/asr-spike/.venv/Scripts/python.exe"

if (-not (Test-Path $py)) {
    Write-Host "ASR venv not found at $py" -ForegroundColor Red
    Write-Host "Create it:  python -m venv E:/asr-spike/.venv" -ForegroundColor Yellow
    Write-Host "  then:     pip install nemo_toolkit[asr] soundfile" -ForegroundColor Yellow
    exit 1
}

$nectecNemo = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"
if ($Model -eq "nectec") {
    if (-not (Test-Path $nectecNemo)) {
        Write-Host "NECTEC checkpoint not found at $nectecNemo -- downloading (~460 MB) ..." -ForegroundColor Yellow
        & $py -c "from huggingface_hub import hf_hub_download as d; d('nectec/Pathumma-stt-th-fastconformer-rnnt-large','pathumma-stt-th-fastconformer-rnnt-large.nemo',local_dir='E:/asr-spike/models')"
    }
    $env:ASR_MODEL = $nectecNemo
    # beam 5 measured best for this model (CER 0.0676 vs 0.0695 greedy) at ~4 ms more per turn
    if (-not $PSBoundParameters.ContainsKey("Strategy")) { $Strategy = "malsd_batch" }
    if (-not $PSBoundParameters.ContainsKey("Beam")) { $Beam = 5 }
} elseif ($Model -eq "typhoon") {
    $env:ASR_MODEL = "typhoon-ai/typhoon-asr-realtime"
} else {
    $env:ASR_MODEL = $Model
}

$env:ASR_PORT = $Port
$env:HF_HOME = "E:/voxcpm-spike/hf-cache"
$env:ASR_DEVICE = $Device
$env:ASR_STRATEGY = $Strategy
$env:ASR_BEAM_SIZE = $Beam
$env:ASR_NGRAM_LM = $NgramLm
$env:ASR_NGRAM_ALPHA = $Alpha
$env:ASR_BOOST_FILE = $BoostFile
$env:ASR_BOOST_ALPHA = $BoostAlpha

Write-Host "Thai ASR [$Model] ($Device, $Strategy) -> http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  POST /asr?sr=16000  body = raw PCM16 mono -> {text, infer_ms}" -ForegroundColor DarkGray
Write-Host "  GET  /health        -> the live decoding config" -ForegroundColor DarkGray
Write-Host "  boot is ~30-60s: import, checkpoint load, warmup" -ForegroundColor DarkGray
if ($NgramLm) {
    Write-Host "  n-gram fusion is ON -- CUDA graphs go off (no Triton on Windows), RTF ~0.13" -ForegroundColor Yellow
}

& $py "$here/server.py"
