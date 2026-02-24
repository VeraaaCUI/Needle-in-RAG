# Full CR-RAG pipeline runner (all splits) with CRRAGModel=local mistral7b (Transformers)
# - Keeps your runtime knobs:
#   * --max-hits 3
#   * $env:CRRAG_OPENAI_CONCURRENCY = "4" (harmless for local HF models)
#   * $env:CRRAG_MAX_CONTEXT_CHARS  = "2000" (used by patched models.py; harmless otherwise)

param(
  [int]$K = 50,
  [int]$KUse = 10,
  [int]$Limit = 50,
  [int]$MaxHits = 3,

  [string]$VictimBackend = "ollama",
  [string]$VictimModel = "gemma3:latest",
  [string]$OllamaUrl = "http://127.0.0.1:11434",

  [string]$CRRAGModel = "mistral7b",
  [string]$CRRAGDevice = "cuda",

  [string]$Defense = "keyword",
  [double]$Alpha = 0.3,
  [double]$Beta = 3.0,
  [double]$Eta = 0.0,
  [int]$SubsampleIter = 1,

  [int]$VictimMaxTokens = 32,
  [int]$CRRAGMaxOutputTokens = 32,

  [switch]$UseCache,
  [string]$CacheDir = "artifacts/crrag_cache"
)

# ---- knobs you said you want to keep (safe even when CRRAGModel is local) ----
if (-not $env:CRRAG_OPENAI_CONCURRENCY) { $env:CRRAG_OPENAI_CONCURRENCY = "4" }
if (-not $env:CRRAG_MAX_CONTEXT_CHARS)  { $env:CRRAG_MAX_CONTEXT_CHARS  = "2000" }

# ---- basic output dirs ----
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
if ($UseCache) { New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null }

if ($VictimBackend -ne "ollama") {
  throw "This rollback script is written for VictimBackend=ollama. (Your current request was only about rolling CRRAG back to local mistral.)"
}

foreach ($dataset in @("nq","msmarco")) {
  $base = Join-Path "data_splits" $dataset
  if (-not (Test-Path $base)) { continue }

  Get-ChildItem -Path $base -Directory | ForEach-Object {
    $splitDir  = $_.FullName
    $splitName = $_.Name

    $answers = Join-Path $splitDir "answers.jsonl"
    $chunks  = Join-Path $splitDir "chunks.jsonl"
    if (-not (Test-Path $answers)) { return }
    if (-not (Test-Path $chunks))  { return }

    Write-Host "`n==== Dataset=$dataset Split=$splitName Victim=$VictimBackend/$VictimModel CRRAG=$CRRAGModel ($CRRAGDevice) ===="

    $indexOut = Join-Path "artifacts" ("{0}_{1}_tfidf" -f $dataset, $splitName)
    $victimSlug = ($VictimModel -replace "[:/\\]","_")

    $run0 = Join-Path "runs" ("{0}_{1}_pass0_limit{2}_{3}_{4}.jsonl" -f $dataset, $splitName, $Limit, $VictimBackend, $victimSlug)
    $met0 = Join-Path "runs" ("{0}_{1}_pass0_limit{2}_{3}_{4}_metrics_evt.json" -f $dataset, $splitName, $Limit, $VictimBackend, $victimSlug)

    $defRun = Join-Path "runs" ("{0}_{1}_crrag_{2}_defended_limit{3}_{4}.jsonl" -f $dataset, $splitName, $Defense, $Limit, $CRRAGModel)
    $defMet = Join-Path "runs" ("{0}_{1}_crrag_{2}_defended_limit{3}_{4}_metrics_evt.json" -f $dataset, $splitName, $Defense, $Limit, $CRRAGModel)

    $rfPass1 = Join-Path "runs" ("{0}_{1}_crrag_{2}_pass1_limit{3}_{4}.jsonl" -f $dataset, $splitName, $Defense, $Limit, $CRRAGModel)
    $rfMet1  = Join-Path "runs" ("{0}_{1}_crrag_{2}_pass1_limit{3}_{4}_pass1_metrics.json" -f $dataset, $splitName, $Defense, $Limit, $CRRAGModel)
    $rfCsv1  = Join-Path "runs" ("{0}_{1}_crrag_{2}_pass1_limit{3}_{4}_per_example.csv" -f $dataset, $splitName, $Defense, $Limit, $CRRAGModel)

    # 1) Index
    python scripts/01_build_index.py --dataset $dataset --chunks $chunks --out $indexOut

    # 2) RAG pass0 (Victim)
    python scripts/02_run_rag.py `
      --dataset $dataset `
      --answers $answers `
      --index $indexOut `
      --k $K `
      --k-use $KUse `
      --limit $Limit `
      --llm-backend ollama `
      --ollama-model $VictimModel `
      --ollama-url $OllamaUrl `
      --max-tokens $VictimMaxTokens `
      --out $run0

    # 3) EVT eval pass0
    python scripts/03_eval_asr.py --run $run0 --out $met0 --tb-k 5

    # 4) CR-RAG adapter (defense + pass1-like)
    $cacheArgs = @()
    if ($UseCache) {
      $cacheArgs += "--use-cache"
      $cacheArgs += "--cache-dir"
      $cacheArgs += $CacheDir
    }

    python scripts/12_crrag_adapter.py `
      --run $run0 `
      --out-run $defRun `
      --out-pass1 $rfPass1 `
      --defense $Defense `
      --alpha $Alpha `
      --beta $Beta `
      --eta $Eta `
      --subsample-iter $SubsampleIter `
      --crrag-model $CRRAGModel `
      --crrag-device $CRRAGDevice `
      --max-hits $MaxHits `
      --max-output-tokens $CRRAGMaxOutputTokens `
      --selected-chunk-mode rank0 `
      --span-mode full_chunk `
      @cacheArgs

    # 5) EVT eval defended
    python scripts/03_eval_asr.py --run $defRun --out $defMet --tb-k 5

    # 6) Pass1 eval
    python scripts/07_eval_pass1.py `
      --pass1 $rfPass1 `
      --chunks $chunks `
      --tb-k 5 `
      --cand-k 5 `
      --out $rfMet1 `
      --out-csv $rfCsv1
  }
}
