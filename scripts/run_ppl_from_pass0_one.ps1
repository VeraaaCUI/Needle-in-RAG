# Run PPL defense baseline starting from an **existing pass0 run file**.
#
# Use-case:
#   You already ran scripts/02_run_rag.py (pass0) earlier (possibly expensive for OpenAI models),
#   and you only want to run:
#     13_ppl_defense -> 07_eval_pass1
#
# This keeps output formats identical to the full pipeline so downstream eval works.

param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [Parameter(Mandatory=$true)][string]$SplitName,

  # Pass0 run selection
  [int]$Limit = 500,
  [ValidateSet('ollama','openai')][string]$VictimBackend = 'openai',
  [string]$VictimModel = 'gpt-5-mini',
  [string]$Pass0Run = '',

  # RAG prompt used-hits (must match the pass0 run's k-use for fair comparison)
  [int]$KUse = 10,

  # Judge model (always local Ollama)
  [string]$OllamaUrl = 'http://127.0.0.1:11434',
  [string]$JudgeModel = 'llama3.1:8b',
  [string]$JudgeOllamaUrl = '',
  [int]$JudgeMaxGenTokens = 128,
  [switch]$NoJudgeWarmup,

  # PPL defense controls
  [int]$MaxJudgeChars = 512,

  # Eval controls
  [int]$TbK = 5,
  [int]$CandK = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Die([string]$msg) {
  throw $msg
}

function Resolve-Pass0Run([string]$dataset, [string]$splitName, [int]$limit, [string]$victimBackend, [string]$victimModel, [string]$explicitPath) {
  if ($explicitPath) {
    if (-not (Test-Path $explicitPath)) {
      Die "Pass0Run path does not exist: $explicitPath"
    }
    return (Resolve-Path $explicitPath).Path
  }

  if (-not (Test-Path 'runs')) {
    Die "Missing runs/ directory. Cannot auto-resolve pass0 run."
  }

  $model_s = ($victimModel -replace "[:/\\]","_")
  $cands = @(
    # Current canonical naming (our runner scripts)
    (Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}_{4}.jsonl" -f $dataset, $splitName, $limit, $victimBackend, $model_s)),
    # Some older runs omitted backend
    (Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}.jsonl" -f $dataset, $splitName, $limit, $model_s)),
    # User-reported naming (contains pass1 but is actually pass0 output)
    (Join-Path 'runs' ("{0}_{1}_pass1_limit{2}_{3}.jsonl" -f $dataset, $splitName, $limit, $model_s))
  )

  foreach ($p in $cands) {
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }

  # Fallback: search by glob, prefer latest modified.
  $glob = "{0}_{1}_*{2}*.jsonl" -f $dataset, $splitName, $model_s
  $hits = Get-ChildItem -Path 'runs' -Filter $glob -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch '_ppl_pass1_' } |
    Where-Object { $_.Name -notmatch '_metrics' } |
    Where-Object { $_.Name -notmatch '_per_example' }

  if ($limit -gt 0) {
    $hits = $hits | Where-Object { $_.Name -match ("limit{0}" -f $limit) }
  }

  $hits = $hits | Sort-Object LastWriteTime -Descending
  if ($hits -and $hits.Count -gt 0) {
    return $hits[0].FullName
  }

  $attempts = ($cands | ForEach-Object { "  - $_" }) -join "`n"
  Die ("Cannot find existing pass0 run JSONL for Dataset=$dataset Split=$splitName Victim=$victimBackend/$victimModel Limit=$limit. Tried:`n$attempts`nAlso searched glob: runs/$glob")
}

# Ensure output dirs exist
New-Item -ItemType Directory -Force -Path 'runs' | Out-Null

# Judge URL defaults to the same OllamaUrl if not provided
if (-not $JudgeOllamaUrl) { $JudgeOllamaUrl = $OllamaUrl }

$splitDir = Join-Path (Join-Path 'data_splits' $Dataset) $SplitName
$chunks   = Join-Path $splitDir 'chunks.jsonl'
if (-not (Test-Path $chunks))  { Die "Missing chunks.jsonl: $chunks" }

$run0 = Resolve-Pass0Run $Dataset $SplitName $Limit $VictimBackend $VictimModel $Pass0Run

Write-Host "`n==== PPL Baseline (FROM PASS0) Dataset=$Dataset Split=$SplitName Victim=$VictimBackend/$VictimModel Judge=ollama/$JudgeModel ===="
Write-Host "Using existing pass0 run: $run0" -ForegroundColor DarkGray

# Normalize model string for filenames
$victim_model_s = ($VictimModel -replace "[:/\\]","_")
$judge_model_s  = ($JudgeModel  -replace "[:/\\]","_")

$pplPass1 = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}.jsonl" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)
$pplMet1  = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}_metrics.json" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)
$pplCsv1  = Join-Path 'runs' ("{0}_{1}_ppl_pass1_limit{2}_{3}_{4}_judge_{5}_per_example.csv" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victim_model_s, $judge_model_s)

# Overwrite only PPL outputs (never touch $run0)
foreach ($p in @($pplPass1,$pplMet1,$pplCsv1)) {
  if (Test-Path $p) { Remove-Item -Force $p }
}

# 13) PPL defense (produces pass1-compatible JSONL)
$maybeWarm = @()
if ($NoJudgeWarmup) { $maybeWarm += '--no-judge-warmup' }

python scripts/13_ppl_defense.py `
  --run $run0 `
  --chunks $chunks `
  --out-pass1 $pplPass1 `
  --judge-backend ollama `
  --judge-model $JudgeModel `
  --ollama-url $JudgeOllamaUrl `
  --judge-max-gen-tokens $JudgeMaxGenTokens `
  --max-judge-chars $MaxJudgeChars `
  --max-hits $KUse `
  --cand-k $CandK `
  @maybeWarm
if ($LASTEXITCODE -ne 0) { Die "Step failed (13_ppl_defense) with exit code $LASTEXITCODE" }

# 07) Reuse pass1 evaluator (chunk-level span evaluation; PPL spans are whole chunks)
python scripts/07_eval_pass1.py `
  --pass1 $pplPass1 `
  --chunks $chunks `
  --tb-k $TbK `
  --cand-k $CandK `
  --out $pplMet1 `
  --out-csv $pplCsv1
if ($LASTEXITCODE -ne 0) { Die "Step failed (07_eval_pass1) with exit code $LASTEXITCODE" }

Write-Host "Done. Outputs:" -ForegroundColor Green
Write-Host "  pass0 run (input): $run0"
Write-Host "  PPL pass1:         $pplPass1"
Write-Host "  PPL metrics:       $pplMet1"
Write-Host "  PPL csv:           $pplCsv1"
