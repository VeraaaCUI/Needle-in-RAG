# Run PPL defense baseline on a single dataset + split.
# Pipeline:
#   01_build_index -> 02_run_rag(pass0) -> 03_eval_asr -> 13_ppl_defense -> 07_eval_pass1

# IMPORTANT (PowerShell): the script-level param() block MUST be the first non-comment statement.
param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [Parameter(Mandatory=$true)][string]$SplitName,

  [int]$K = 200,
  [int]$KUse = 10,
  [int]$Limit = 500,

  [ValidateSet('ollama','openai')][string]$VictimBackend = 'ollama',
  [string]$VictimModel = 'gemma:7b',

  # Victim backend endpoints
  [string]$OllamaUrl = 'http://127.0.0.1:11434',
  [string]$OpenAIBaseUrl = '',
  [string]$OpenAIApiKeyEnv = '',

  # Judge model (always local Ollama in this baseline)
  [ValidateSet('ollama')][string]$JudgeBackend = 'ollama',
  [string]$JudgeModel = 'llama3.1:8b',
  [string]$JudgeOllamaUrl = '',

  # Speed/robustness controls for the Ollama judge
  [int]$JudgeMaxGenTokens = 128,
  [switch]$NoJudgeWarmup,

  # Eval controls
  [int]$TbK = 5,
  [int]$CandK = 5,

  # PPL defense controls
  [int]$MaxJudgeChars = 512,

  # Victim gen controls
  [int]$MaxTokens = 256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Ok([int]$code, [string]$step) {
  if ($code -ne 0) {
    throw "Step failed ($step) with exit code $code"
  }
}

# Ensure output dirs exist
New-Item -ItemType Directory -Force -Path 'runs' | Out-Null
New-Item -ItemType Directory -Force -Path 'artifacts' | Out-Null

# Judge URL defaults to the same OllamaUrl if not provided
if (-not $JudgeOllamaUrl) { $JudgeOllamaUrl = $OllamaUrl }

# Copy alternative env-var into OPENAI_API_KEY if requested
if ($OpenAIApiKeyEnv -and -not $env:OPENAI_API_KEY) {
  $k = [Environment]::GetEnvironmentVariable($OpenAIApiKeyEnv)
  if ($k) { $env:OPENAI_API_KEY = $k }
}
if ($OpenAIBaseUrl) {
  $env:OPENAI_BASE_URL = $OpenAIBaseUrl
}

$splitDir = Join-Path (Join-Path 'data_splits' $Dataset) $SplitName
$answers  = Join-Path $splitDir 'answers.jsonl'
$chunks   = Join-Path $splitDir 'chunks.jsonl'
if (-not (Test-Path $answers)) { throw "Missing answers.jsonl: $answers" }
if (-not (Test-Path $chunks))  { throw "Missing chunks.jsonl: $chunks" }

Write-Host "`n==== PPL Baseline Dataset=$Dataset Split=$SplitName Victim=$VictimBackend/$VictimModel Judge=$JudgeBackend/$JudgeModel ===="

$indexOut = Join-Path 'artifacts' ("{0}_{1}_tfidf" -f $Dataset, $SplitName)

# Normalize model string for filenames
$victim_model_s = ($VictimModel -replace "[:/\\]","_")
$judge_model_s  = ($JudgeModel  -replace "[:/\\]","_")

$run0 = Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}_{4}.jsonl" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s)
$met0 = Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}_{4}_metrics_evt.json" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s)

$pplPass1 = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}.jsonl" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)
$pplMet1  = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}_metrics.json" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)
$pplCsv1  = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}_per_example.csv" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)

# Overwrite outputs to keep runs deterministic
foreach ($p in @($run0,$met0,$pplPass1,$pplMet1,$pplCsv1)) {
  if (Test-Path $p) { Remove-Item -Force $p }
}

# 01) Build TF-IDF index (deterministic)
python scripts/01_build_index.py --dataset $Dataset --chunks $chunks --out $indexOut
Assert-Ok $LASTEXITCODE '01_build_index'

# 02) Run pass0 RAG
if ($VictimBackend -eq 'ollama') {
  python scripts/02_run_rag.py `
    --dataset $Dataset `
    --answers $answers `
    --index $indexOut `
    --k $K `
    --k-use $KUse `
    --limit $Limit `
    --llm-backend ollama `
    --ollama-model $VictimModel `
    --ollama-url $OllamaUrl `
    --max-tokens $MaxTokens `
    --out $run0
} else {
  $args = @(
    "scripts/02_run_rag.py",
    "--dataset", $Dataset,
    "--answers", $answers,
    "--index", $indexOut,
    "--k", $K,
    "--k-use", $KUse,
    "--limit", $Limit,
    "--llm-backend", "openai",
    "--openai-model", $VictimModel,
    "--max-tokens", $MaxTokens,
    "--out", $run0
  )
  if ($OpenAIBaseUrl)   { $args += @("--openai-base-url", $OpenAIBaseUrl) }
  if ($OpenAIApiKeyEnv) { $args += @("--openai-api-key-env", $OpenAIApiKeyEnv) }
  python @args
}
Assert-Ok $LASTEXITCODE '02_run_rag(pass0)'

# 03) Eval pass0 EVT metrics
python scripts/03_eval_asr.py --run $run0 --out $met0 --tb-k $TbK
Assert-Ok $LASTEXITCODE '03_eval_asr(pass0)'

# 13) PPL defense (produces pass1-compatible JSONL)
$maybeWarm = @()
if ($NoJudgeWarmup) { $maybeWarm += '--no-judge-warmup' }

python scripts/13_ppl_defense.py `
  --run $run0 `
  --chunks $chunks `
  --out-pass1 $pplPass1 `
  --judge-backend $JudgeBackend `
  --judge-model $JudgeModel `
  --ollama-url $JudgeOllamaUrl `
  --judge-max-gen-tokens $JudgeMaxGenTokens `
  --max-judge-chars $MaxJudgeChars `
  --max-hits $KUse `
  --cand-k $CandK `
  @maybeWarm
Assert-Ok $LASTEXITCODE '13_ppl_defense'

# 07) Reuse pass1 evaluator (chunk-level span evaluation; PPL spans are whole chunks)
python scripts/07_eval_pass1.py `
  --pass1 $pplPass1 `
  --chunks $chunks `
  --tb-k $TbK `
  --cand-k $CandK `
  --out $pplMet1 `
  --out-csv $pplCsv1
Assert-Ok $LASTEXITCODE '07_eval_pass1'

Write-Host "Done. Outputs:" -ForegroundColor Green
Write-Host "  pass0 run:   $run0"
Write-Host "  pass0 evt:   $met0"
Write-Host "  PPL pass1:   $pplPass1"
Write-Host "  PPL metrics: $pplMet1"
Write-Host "  PPL csv:     $pplCsv1"
