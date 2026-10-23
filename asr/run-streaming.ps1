#requires -Version 5.1
# Start the STREAMING Thai ASR server (GPU). Runs in the ASR venv -- NOT the VoxCPM one.
#
# This is the cache-aware model that emits partial transcripts WHILE the caller is talking.
# It does NOT replace .\run.ps1 (:8007), which is more accurate and still owns the final
# transcript -- see the header of asr/server_streaming.py for the measured comparison.
#
#   .\run-streaming.ps1                 -> port 8011, 240 ms look-ahead, GPU
#   .\run-streaming.ps1 -Right 13       -> 1040 ms look-ahead (measured: no better)
#   .\run-streaming.ps1 -Device cpu     -> untested; measure before trusting it
param(
    [int]$Port = 8011,
    [ValidateSet(3, 6, 13)][int]$Right = 3,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "E:/asr-spike/.venv/Scripts/python.exe"

if (-not (Test-Path $py)) {
    Write-Host "ASR venv not found at $py" -ForegroundColor Red
    exit 1
}

# Deliberately NOT setting HF_HOME: the 1.6 GB checkpoint already sits in the default cache
# at ~/.cache/huggingface. run.ps1 repoints HF_HOME at E:/voxcpm-spike/hf-cache, which would
# force a second download of the same file.
$env:ASR_STREAM_PORT = $Port
$env:ASR_STREAM_RIGHT = $Right
$env:ASR_STREAM_DEVICE = $Device

Write-Host "Typhoon ASR STREAMING ($Device) -> http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  POST /stream?sid=N&final=0  body = raw PCM16 mono -> {text}" -ForegroundColor DarkGray
Write-Host "  look-ahead: $($Right * 80) ms" -ForegroundColor DarkGray
Write-Host "  boot is ~25-40s: import, checkpoint restore, warmup" -ForegroundColor DarkGray

& $py "$here/server_streaming.py"
