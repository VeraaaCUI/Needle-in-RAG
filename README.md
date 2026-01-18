# RAG Char-Trace (Poisoning Reproduction Scaffold)

本仓库提供一个**可复现实验记录（trace）**的 RAG 框架，用于在**已污染的检索库（poisoned corpus）**上复现前人工作中的“中毒效果”（例如 ASR/攻击成功率），并为后续“字符级（span-level）查毒/归因”研究预留接口。

你当前已经具备的数据格式（示例）：
- `data/<dataset>/chunks.jsonl`：每行一个 chunk，包含 `chunk_id`, `text`, `guilty_spans`, `meta` 等
- `data/<dataset>/answers.jsonl`：每行一个问题样本，包含 `qid`, `question`, `answer`, `incorrect_answer` 等

## 1. 目录结构

```
rag_char_trace/
  src/rag_char_trace/
    data/            # JSONL 读取与数据结构
    index/           # TF-IDF 索引与检索
    llm/             # LLM 后端（Ollama / llama.cpp / OpenAI API）
    rag/             # RAG pipeline
    trace/           # 统一 trace 日志
    eval/            # ASR/EM 评测
    utils/           # 文本归一化等
  scripts/           # 可执行脚本（PowerShell 友好）
  configs/           # YAML 配置
  artifacts/         # 索引等中间产物（默认不提交）
  runs/              # RAG 运行日志与评测输出（默认不提交）
  data/              # 放置你的数据（不提交）
```

## 2. 环境搭建（Windows PowerShell）

建议 Python 3.10+。

```powershell
cd rag_char_trace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### 2.1 本地小模型优先（推荐起步：Ollama）

最省事的方式是用 Ollama 在本地跑 7B/8B，例如 `llama3:8b` 或 `llama2:7b`。

- 安装并启动 Ollama 后，拉取模型：

```powershell
ollama pull llama3:8b
```

本仓库的 `ollama` 后端通过 HTTP 调用本地 Ollama 服务。

### 2.2 OpenAI API（后续扩展）

设置环境变量：

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

然后在脚本参数中选择 `--llm-backend openai`。

## 3. 快速复现实验：Build Index -> Run RAG -> Evaluate ASR

### 3.1 构建 TF-IDF 索引

```powershell
python scripts/01_build_index.py --dataset nq --chunks data/nq/chunks.jsonl --out artifacts/nq_tfidf
```

### 3.2 运行 RAG（Ollama / Llama 7B/8B）

```powershell
python scripts/02_run_rag.py `
  --dataset nq `
  --answers data/nq/answers.jsonl `
  --index artifacts/nq_tfidf `
  --k 10 `
  --llm-backend ollama `
  --ollama-model llama3:8b `
  --out runs/nq_llama3_8b_k10.jsonl
```

该脚本会输出每个问题的：
- query
- top-k 检索到的 chunks（含 `chunk_id`, `meta.poisoned`, `text`）
- 组合后的 prompt
- 模型输出与运行耗时

### 3.3 评测 ASR（攻击成功率）

```powershell
python scripts/03_eval_asr.py --run runs/nq_llama3_8b_k10.jsonl --answers data/nq/answers.jsonl --out runs/nq_llama3_8b_k10_metrics.json
```

默认 ASR 判定：预测答案（归一化后）是否匹配 `incorrect_answer`。

## 4. Trace 设计（为后续字符级查毒做准备）

本仓库的 `trace` 记录以 JSONL 输出，每条样本包含：
- `retrieval`: top-k chunks + 分数
- `generation`: prompt + raw completion
- `meta`: dataset/attack/qid 等

后续做字符级查毒时，可以在 `trace` 中加入：
- `suspicious_spans`: 对检索 chunk 的可疑字符片段（start/end）
- `attribution`: “答案片段 -> 证据 chunk -> span” 的映射

## 5. 与你给出的数据字段如何对齐

- `chunks.jsonl` 中的 `guilty_spans` 目前仅用于**上帝视角**评估（可选）；默认 pipeline 不依赖它。
- `meta.poisoned=true` 用于统计与可视化（例如检索到的 top-k 中毒比例）。

## 6. 下一步建议（面向发表）

1) 在不同 `k`、不同 retriever（TF-IDF/BM25/向量检索）下复现 ASR 表格。
2) 固化 trace 与随机种子，保证实验可复现。
3) 基于 trace 扩展字符级归因（span-level），并将“查毒/解毒策略”与 ASR 降低幅度关联。


## 7. 复现类似 Table-2 的按 dataset/attack 分组 ASR 表

如果你的 `answers.jsonl` 里已经包含 `dataset` 与 `attack` 字段（你给的示例就是如此），那么单次 `run.jsonl` 里会保留这些 meta 字段。
你可以直接对一次运行结果做 **(dataset, attack)** 分组统计，并导出 CSV/LaTeX：

```powershell
python scripts/05_group_asr_table.py `
  --run runs/nq_llama3_8b_k10.jsonl `
  --out-json runs/nq_grouped_asr.json `
  --out-csv runs/nq_grouped_asr.csv `
  --out-tex runs/nq_grouped_asr.tex
```

## 7. 导出“Dataset x Attack”的 ASR 表（更接近论文 Table 2）

如果你的 `answers.jsonl`/`meta` 字段中包含 `dataset` 与 `attack`，且你一次性跑了一个包含多种 attack 的 run（或把多个 run 合并到一个 JSONL），可以用下面脚本直接导出宽表：

```powershell
python scripts/05_group_asr_table.py --run runs/nq_llama3_8b_k10.jsonl --out-csv runs/table2.csv --out-tex runs/table2.tex
```

说明：该脚本默认从 `run.jsonl` 里的 `meta.dataset` 与 `meta.attack` 做分组；若缺失则会显示为 `(unknown)`。

### 3.2.1 大 K (200/500) 的现实限制与推荐参数

当你将 `k` 设为 200/500 时，直接把 top-k 全量拼进 prompt 往往会超出小模型 (7B/8B) 的上下文窗口。
本仓库在 `scripts/02_run_rag.py` 提供了三个参数用于控制 prompt 大小：

- `--prompt-chunk-chars`：每个 chunk 放进 prompt 的最大字符数（默认 1000）
- `--max-context-chars`：所有上下文合计最大字符数（默认 12000）
- `--trace-chunk-chars`：写入 run JSONL 时保存的 chunk 文本预览长度（默认 512；完整文本可由 chunk_id 回查）

例如：

```powershell
python scripts/02_run_rag.py `
  --dataset nq `
  --answers data/nq/answers.jsonl `
  --index artifacts/nq_tfidf `
  --k 500 `
  --prompt-chunk-chars 400 `
  --max-context-chars 12000 `
  --llm-backend ollama `
  --ollama-model llama3:8b `
  --out runs/nq_llama3_8b_k500.jsonl
```
