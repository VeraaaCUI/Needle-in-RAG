# Run PPL_Character baseline on one dataset/split.
# Pipeline:
#   01_build_index -> 02_run_rag(pass0) -> 03_eval_asr -> 14_ppl_character -> 07_eval_pass1

param(
  [Parameter(Mandatory=$true)][string]$Dataset,
  [Parameter(Mandatory=$true)][string]$SplitName,

  [int]$K = 200,
  [int]$KUse = 10,
  [int]$Limit = 100,

  [ValidateSet('ollama','openai','llama_cpp')][string]$VictimBackend = 'ollama',
  [string]$VictimModel = 'llama3:8b',
  [string]$OllamaUrl = 'http://127.0.0.1:11434',

  # OpenAI routing (only used when VictimBackend=openai)
  [string]$OpenAIBaseUrl = $null,
  [string]$OpenAIApiKeyEnv = $null,

  # Judge (PPL)
  [ValidateSet('ollama','hf')][string]$JudgeBackend = 'ollama',
  [string]$JudgeModel = 'llama3.1:8b',
  [string]$HFJudgeModel = 'gpt2',
  [string]$HFDevice = 'cpu',

  # PPL parameters
  [int]$MaxJudgeChars = 800,
  [int]$MaxLocChars = 1200,
  [int]$WindowChars = 256,
  [int]$MinWindowChars = 64,
  [int]$MaxWindowsPerLevel = 24,
  [int]$TopKWindows = 10,
  [string]$CacheDir = $null,

  # Evaluation
  [int]$TbK = 5,
  [int]$CandK = 5,

  # Victim generation
  [int]$MaxTokens = 32
)

$splitDir = Join-Path (Join-Path 'data_splits' $Dataset) $SplitName
$answers  = Join-Path $splitDir 'answers.jsonl'
$chunks   = Join-Path $splitDir 'chunks.jsonl'

if (-not (Test-Path $answers)) { throw "Missing answers.jsonl: $answers" }
if (-not (Test-Path $chunks))  { throw "Missing chunks.jsonl:  $chunks" }

New-Item -ItemType Directory -Force -Path 'artifacts' | Out-Null
New-Item -ItemType Directory -Force -Path 'runs' | Out-Null

$victimTag = ($VictimModel -replace '[:/\\]','_')
$judgeTag  = ($JudgeModel -replace '[:/\\]','_')

$indexOut = Join-Path 'artifacts' ("{0}_{1}_tfidf" -f $Dataset, $SplitName)

$run0 = Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}_{4}.jsonl" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victimTag)
$met0 = Join-Path 'runs' ("{0}_{1}_pass0_limit{2}_{3}_{4}_metrics_evt.json" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victimTag)

$pplCharPass1 = Join-Path 'runs' ("{0}_{1}_ppl_character_pass1_limit{2}_{3}_{4}_judge_{5}.jsonl" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victimTag, $judgeTag)

$pplCharMet1  = Join-Path 'runs' ("{0}_{1}_ppl_character_pass1_limit{2}_{3}_{4}_judge_{5}_pass1_metrics.json" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victimTag, $judgeTag)
$pplCharCsv1  = Join-Path 'runs' ("{0}_{1}_ppl_character_pass1_limit{2}_{3}_{4}_judge_{5}_per_example.csv" -f $Dataset, $SplitName, $Limit, $VictimBackend, $victimTag, $judgeTag)

Write-Host "`n==== PPL_Character Dataset=$Dataset Split=$SplitName Victim=$VictimBackend/$VictimModel Judge=$JudgeBackend/$JudgeModel ===="

# 01) Build index
python scripts/01_build_index.py --dataset $Dataset --chunks $chunks --out $indexOut

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
}
elseif ($VictimBackend -eq 'openai') {
  $openaiArgs = @()
  if ($OpenAIBaseUrl)    { $openaiArgs += @('--openai-base-url', $OpenAIBaseUrl) }
  if ($OpenAIApiKeyEnv)  { $openaiArgs += @('--openai-api-key-env', $OpenAIApiKeyEnv) }

  python scripts/02_run_rag.py `
    --dataset $Dataset `
    --answers $answers `
    --index $indexOut `
    --k $K `
    --k-use $KUse `
    --limit $Limit `
    --llm-backend openai `
    --openai-model $VictimModel `
    --max-tokens $MaxTokens `
    --out $run0 @openaiArgs
}
else {
  throw "VictimBackend '$VictimBackend' not supported by this runner yet."
}

# 03) Eval pass0
python scripts/03_eval_asr.py --run $run0 --out $met0 --tb-k $TbK

# 14) PPL_Character baseline (writes a pass1-compatible JSONL)
$judgeArgs = @()
if ($JudgeBackend -eq 'ollama') {
  $judgeArgs += @('--judge-backend','ollama','--judge-model',$JudgeModel,'--ollama-url',$OllamaUrl)
}
else {
  $judgeArgs += @('--judge-backend','hf','--hf-model',$HFJudgeModel,'--hf-device',$HFDevice)
}

$cacheArgs = @()
if ($CacheDir) {
  $cacheArgs += @('--cache-dir', $CacheDir)
}

python scripts/14_ppl_character.py `
  --run $run0 `
  --chunks $chunks `
  --out-pass1 $pplCharPass1 `
  --max-hits $KUse `
  --max-judge-chars $MaxJudgeChars `
  --max-loc-chars $MaxLocChars `
  --window-chars $WindowChars `
  --min-window-chars $MinWindowChars `
  --max-windows-per-level $MaxWindowsPerLevel `
  --topk-windows $TopKWindows @judgeArgs @cacheArgs

# 07) Eval pass1 metrics on the baseline's output
python scripts/07_eval_pass1.py `
  --pass1 $pplCharPass1 `
  --chunks $chunks `
  --tb-k $TbK `
  --cand-k $CandK `
  --out $pplCharMet1 `
  --out-csv $pplCharCsv1

Write-Host "Done.\n  pass0: $run0\n  ppl_character_pass1: $pplCharPass1\n  metrics: $pplCharMet1"
