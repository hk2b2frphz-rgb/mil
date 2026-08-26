param(
    [string]$InputJsonl = "",
    [string]$OutputDir = "",
    [Nullable[int]]$NumDialogues = $null
)

$ErrorActionPreference = "Stop"
$PocRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $PocRoot
if (-not $InputJsonl) { $InputJsonl = Join-Path $PocRoot "artifacts/train_dialogues.jsonl" }
if (-not $OutputDir) { $OutputDir = Join-Path $PocRoot "artifacts/tts" }
if (-not (Test-Path -LiteralPath $InputJsonl -PathType Leaf)) {
    throw "Dialogue JSONL not found: $InputJsonl"
}

$Arguments = @(
    "run", "python", "scripts/generate_qwen3_tts_data.py",
    "--out-dir", (Join-Path $OutputDir "training_set"),
    "--dialogues-jsonl", $InputJsonl,
    "--model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "--speaker-moshi", "Ono_Anna",
    "--speaker-user", "Serena",
    "--user-speaker-pool", "Serena,Sohee,Vivian,Dylan,Eric,Aiden,Uncle_Fu,Ryan",
    "--speaker-other", "Dylan",
    "--speaker-background", "Ryan",
    "--device", "cuda",
    "--dtype", "float16",
    "--no-opening-greeting",
    "--no-emotion",
    "--whole-utterance"
)
if ($null -ne $NumDialogues) { $Arguments += @("--num-dialogues", "$NumDialogues") }

Push-Location $RepoRoot
try { & uv @Arguments; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
finally { Pop-Location }
