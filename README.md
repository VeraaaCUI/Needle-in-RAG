# RAGCharacter: Character-Level Poison Traceback in RAG

## 📌 Overview

This repository implements **RAGCharacter**, a black-box forensic framework for **character-level poison traceback** in Retrieval-Augmented Generation (RAG) systems.

Retrieval-augmented generation improves factual grounding by conditioning LLMs on external evidence. However, it also introduces a data-layer attack surface: adversaries can inject poisoned corpus entries to steer model outputs without modifying model parameters.

Existing defenses and traceback approaches are mostly passage-level, which is too coarse for modern attacks where the effective payload may be:

- a short fabricated claim
- a trigger phrase
- a hidden instruction embedded in benign text

RAGCharacter addresses this by performing fine-grained attribution at the character/span level.

---

## 📄 Paper Abstract

**Authors:**  
Huining Cui, Wei Liu  
School of Computer Science, University of Technology Sydney, Australia

**Abstract:**

Retrieval-augmented generation (RAG) improves factual grounding by conditioning large language models on retrieved evidence, but it also opens a data-layer attack surface: poisoned corpus entries can steer outputs without changing model parameters. Existing defenses and traceback methods are largely passage-level, which is too coarse for modern attacks whose effective payload may be a short fabricated claim, trigger phrase, or hidden instruction embedded inside an otherwise benign chunk.

We study black-box character-level poison traceback in RAG and present **RAGCharacter**, a two-pass forensic framework that localizes the responsible retrieved span for a concrete misgeneration event.

**Pass-0** runs standard RAG while logging a prompt-anchored execution trace. **Pass-1** re-enters a triggered trace and performs event-conditioned traceback over prompt-used evidence via budgeted counterfactual masking and replay, yielding an attribution span for forensic reporting and a causal span under the logged trace.

We further introduce an evaluation protocol that measures both event-level chunk traceback and character-level localization fidelity.

Across two QA corpora, five poisoning attack families, six target LLMs, and multiple baselines, RAGCharacter achieves the best trade-off between localization accuracy and low over-attribution.

These results suggest that prompt-conditioned, black-box character-level traceback is feasible in closed-source deployment settings, enabling fine-grained evidence auditing and remediation.

---

## 🚀 Key Features

### 🔍 Two-pass traceback framework

- **Pass-0:** standard RAG + trace logging
- **Pass-1:** counterfactual masking + replay

### ✂️ Character-level attribution

- Span localization
- Causal span identification

### 📊 Evaluation metrics

- **EVT**: event-level traceback
- **TB@K / MRR**: retrieval attribution
- **Char-F1 / IoU / FPR**: fine-grained localization

### 🧪 Supports multiple attack settings

- Corpus poisoning
- Prompt injection
- Multi-poison competition

### 🔌 Flexible LLM backends

- Ollama recommended for local
- OpenAI API
- llama.cpp optional

---

## 📁 Project Structure

```text
rag_char_trace/
  src/rag_char_trace/
    data/            # JSONL loaders and data structures
    index/           # TF-IDF indexing and retrieval
    llm/             # LLM backends (Ollama / OpenAI / etc.)
    rag/             # RAG pipeline
    trace/           # Execution trace logging
    eval/            # ASR / EM / traceback evaluation
    utils/           # Text normalization and helpers

  scripts/           # Executable scripts (PowerShell-friendly)
  configs/           # YAML configs
  artifacts/         # Index outputs (not committed)
  runs/              # Logs and metrics (not committed)
  data/              # Dataset directory (not committed)
```

---

## ⚙️ Environment Setup

### Windows PowerShell

Python 3.10+ recommended.

```powershell
cd rag_char_trace

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
```

---

## 🧪 Running the Full Pipeline

### 🔁 Full RAG + Traceback Pipeline

```powershell
.\.venv\Scripts\Activate.ps1

$k    = 200
$kuse = 10
$limit = 500

$model  = "gemma:7b"
$ollama = "http://127.0.0.1:11434"

foreach ($dataset in @("nq","msmarco")) {

  $base = Join-Path "data_splits" $dataset

  Get-ChildItem -Path $base -Directory | ForEach-Object {

    $splitDir  = $_.FullName
    $splitName = $_.Name

    $answers = Join-Path $splitDir "answers.jsonl"
    $chunks  = Join-Path $splitDir "chunks.jsonl"

    $indexOut = Join-Path "artifacts" ("{0}_{1}_tfidf" -f $dataset, $splitName)

    $run0 = Join-Path "runs" ("{0}_{1}_pass0_limit{2}.jsonl" -f $dataset, $splitName, $limit)
    $met0 = Join-Path "runs" ("{0}_{1}_pass0_limit{2}_metrics.json" -f $dataset, $splitName, $limit)

    $run1 = Join-Path "runs" ("{0}_{1}_pass1_limit{2}.jsonl" -f $dataset, $splitName, $limit)
    $met1 = Join-Path "runs" ("{0}_{1}_pass1_limit{2}_pass1_metrics.json" -f $dataset, $splitName, $limit)
    $csv1 = Join-Path "runs" ("{0}_{1}_pass1_limit{2}_metrics_per_example.csv" -f $dataset, $splitName, $limit)

    Write-Host "`n==== Dataset=$dataset Split=$splitName ===="

    python scripts/01_build_index.py --dataset $dataset --chunks $chunks --out $indexOut

    python scripts/02_run_rag.py `
      --dataset $dataset `
      --answers $answers `
      --index $indexOut `
      --k $k `
      --k-use $kuse `
      --limit $limit `
      --llm-backend ollama `
      --ollama-model $model `
      --ollama-url $ollama `
      --temperature 0 `
      --max-tokens 256 `
      --out $run0

    python scripts/03_eval_asr.py --run $run0 --out $met0

    python scripts/06_pass1_traceback.py `
      --run $run0 `
      --out $run1 `
      --trigger PAR `
      --objective event `
      --candidate-mode event `
      --max-rounds 3 `
      --max-chunks-to-test 5 `
      --max-sentences-to-test 5 `
      --top-sentences 2 `
      --max-span-candidates 12 `
      --max-bisect-steps 6 `
      --min-span-len 4 `
      --llm-backend ollama `
      --ollama-model $model `
      --ollama-url $ollama `
      --temperature 0 `
      --max-tokens 64

    python scripts/07_eval_pass1.py `
      --pass1 $run1 `
      --chunks $chunks `
      --tb-k 5 `
      --cand-k 5 `
      --out $met1 `
      --out-csv $csv1
  }
}
```
