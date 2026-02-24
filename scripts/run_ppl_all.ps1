# Traverse all data_splits/nq and data_splits/msmarco splits and run the PPL defense baseline pipeline.
# Pipeline:
#   01_build_index -> 02_run_rag(pass0) -> 03_eval_asr -> 13_ppl_defense -> 07_eval_pass1

# IMPORTANT (PowerShell): the script-level param() block MUST be the first non-comment statement.
param(
  [int]$K = 200,
  [int]$KUse = 10,
  [int]$Limit = 500,

  [ValidateSet('ollama','openai')][string]$VictimBackend = 'ollama',
  [string]$VictimModel = 'gemma:7b',

  # Ollama endpoint (used for VictimBackend=ollama, and also for the Judge model)
  [string]$OllamaUrl = 'http://127.0.0.1:11434',

  # OpenAI routing options for VictimBackend=openai (optional)
  [string]$OpenAIBaseUrl = '',
  [string]$OpenAIApiKeyEnv = '',

  # Judge model (local Ollama)
  [string]$JudgeModel = 'llama3.1:8b',
  [string]$JudgeOllamaUrl = '',

  # Speed/robustness controls for the Ollama judge
  [int]$JudgeMaxGenTokens = 128,
  [switch]$NoJudgeWarmup,

  [int]$TbK = 5,
  [int]$CandK = 5,
  [int]$MaxJudgeChars = 512,

  [int]$MaxTokens = 256,
  [switch]$ContinueOnError = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

foreach ($dataset in @('nq','msmarco')) {
  $base = Join-Path 'data_splits' $dataset
  if (-not (Test-Path $base)) { continue }

  foreach ($dir in (Get-ChildItem -Path $base -Directory)) {
    $splitDir  = $dir.FullName
    $splitName = $dir.Name

    $answers = Join-Path $splitDir 'answers.jsonl'
    $chunks  = Join-Path $splitDir 'chunks.jsonl'
    if (-not (Test-Path $answers)) { continue }
    if (-not (Test-Path $chunks))  { continue }

    try {
      ./scripts/run_ppl_one.ps1 `
        -Dataset $dataset `
        -SplitName $splitName `
        -K $K `
        -KUse $KUse `
        -Limit $Limit `
        -VictimBackend $VictimBackend `
        -VictimModel $VictimModel `
        -OllamaUrl $OllamaUrl `
        -OpenAIBaseUrl $OpenAIBaseUrl `
        -OpenAIApiKeyEnv $OpenAIApiKeyEnv `
        -JudgeModel $JudgeModel `
        -JudgeOllamaUrl $JudgeOllamaUrl `
        -JudgeMaxGenTokens $JudgeMaxGenTokens `
        -NoJudgeWarmup:$NoJudgeWarmup `
        -TbK $TbK `
        -CandK $CandK `
        -MaxJudgeChars $MaxJudgeChars `
        -MaxTokens $MaxTokens
    } catch {
      Write-Host ("!! Failed Dataset={0} Split={1}: {2}" -f $dataset, $splitName, $_.Exception.Message) -ForegroundColor Red
      if (-not $ContinueOnError) { throw }
    }
  }
}
