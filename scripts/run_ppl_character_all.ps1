# Run PPL_Character baseline across all dataset splits under data_splits/<dataset>/*

param(
  [string]$Dataset = 'nq',

  [int]$K = 200,
  [int]$KUse = 10,
  [int]$Limit = 100,

  [ValidateSet('ollama','openai','llama_cpp')][string]$VictimBackend = 'ollama',
  [string]$VictimModel = 'llama3:8b',
  [string]$OllamaUrl = 'http://127.0.0.1:11434',

  # OpenAI routing
  [string]$OpenAIBaseUrl = $null,
  [string]$OpenAIApiKeyEnv = $null,

  # Judge
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

  # Eval
  [int]$TbK = 5,
  [int]$CandK = 5,

  # Victim generation
  [int]$MaxTokens = 32
)

$root = Join-Path 'data_splits' $Dataset
if (-not (Test-Path $root)) { throw "Missing dataset dir: $root" }

$splitDirs = Get-ChildItem -Directory $root | Sort-Object Name
if (-not $splitDirs) { throw "No splits found under: $root" }

foreach ($d in $splitDirs) {
  $splitName = $d.Name
  .\scripts\run_ppl_character_one.ps1 `
    -Dataset $Dataset `
    -SplitName $splitName `
    -K $K `
    -KUse $KUse `
    -Limit $Limit `
    -VictimBackend $VictimBackend `
    -VictimModel $VictimModel `
    -OllamaUrl $OllamaUrl `
    -OpenAIBaseUrl $OpenAIBaseUrl `
    -OpenAIApiKeyEnv $OpenAIApiKeyEnv `
    -JudgeBackend $JudgeBackend `
    -JudgeModel $JudgeModel `
    -HFJudgeModel $HFJudgeModel `
    -HFDevice $HFDevice `
    -MaxJudgeChars $MaxJudgeChars `
    -MaxLocChars $MaxLocChars `
    -WindowChars $WindowChars `
    -MinWindowChars $MinWindowChars `
    -MaxWindowsPerLevel $MaxWindowsPerLevel `
    -TopKWindows $TopKWindows `
    -CacheDir $CacheDir `
    -TbK $TbK `
    -CandK $CandK `
    -MaxTokens $MaxTokens
}
