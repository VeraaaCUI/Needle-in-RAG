# Traverse all data_splits/* splits and run PPL defense baseline
# starting from **existing pass0 run files in runs/**.

param(
  [int]$KUse = 10,
  [int]$Limit = 500,

  [ValidateSet('ollama','openai')][string]$VictimBackend = 'openai',
  [string]$VictimModel = 'gpt-5-mini',

  # Ollama base URL is used for judge; also used by victim backend when VictimBackend=ollama
  [string]$OllamaUrl = 'http://127.0.0.1:11434',

  # Judge model (always local Ollama)
  [string]$JudgeModel = 'llama3.1:8b',
  [string]$JudgeOllamaUrl = '',
  [int]$JudgeMaxGenTokens = 128,
  [int]$MaxJudgeChars = 512,
  [switch]$NoJudgeWarmup,

  # Eval controls
  [int]$TbK = 5,
  [int]$CandK = 5,

  # Behaviour
  [switch]$FailFast
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

foreach ($dataset in @('nq','msmarco')) {
  $base = Join-Path 'data_splits' $dataset
  if (-not (Test-Path $base)) { continue }

  Get-ChildItem -Path $base -Directory | ForEach-Object {
    $splitDir  = $_.FullName
    $splitName = $_.Name

    $chunks  = Join-Path $splitDir 'chunks.jsonl'
    if (-not (Test-Path $chunks)) { return }

    try {
      ./scripts/run_ppl_from_pass0_one.ps1 `
        -Dataset $dataset `
        -SplitName $splitName `
        -Limit $Limit `
        -VictimBackend $VictimBackend `
        -VictimModel $VictimModel `
        -KUse $KUse `
        -OllamaUrl $OllamaUrl `
        -JudgeModel $JudgeModel `
        -JudgeOllamaUrl $JudgeOllamaUrl `
        -JudgeMaxGenTokens $JudgeMaxGenTokens `
        -MaxJudgeChars $MaxJudgeChars `
        -TbK $TbK `
        -CandK $CandK `
        -NoJudgeWarmup:$NoJudgeWarmup
    } catch {
      Write-Warning "!! Skip Dataset=$dataset Split=$splitName : $($_.Exception.Message)"
      if ($FailFast) { throw }
    }
  }
}
