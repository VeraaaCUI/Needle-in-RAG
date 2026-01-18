from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from rag_char_trace.data.io import load_answers
from rag_char_trace.index.tfidf import load_tfidf_index
from rag_char_trace.llm.base import LLMConfig
from rag_char_trace.llm.ollama import OllamaLLM
from rag_char_trace.llm.openai_api import OpenAILLM
from rag_char_trace.rag.pipeline import PromptSpec, run_one
from rag_char_trace.trace.logger import JsonlTraceWriter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-use", type=int, default=10, help="How many retrieved passages to include in the prompt (0 = as many as fit).")
    # Quick experiment: run only the first N QA items.
    # --max-questions is kept for backward compatibility; prefer --limit.
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N QA items")
    ap.add_argument("--max-questions", type=int, default=None, help=argparse.SUPPRESS)

    ap.add_argument("--llm-backend", choices=["ollama", "openai", "llama_cpp"], default="ollama")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=128)

    ap.add_argument("--ollama-model", default="llama3:8b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")

    ap.add_argument("--openai-model", default="gpt-4o-mini")

    ap.add_argument(
        "--system",
        default=(
            "You are a question answering system. You MUST use ONLY the provided passages. "
            "Do NOT use any external knowledge or assumptions. "
            "If the answer is not explicitly stated in the passages, output exactly: UNKNOWN."
        ),
    )
    ap.add_argument(
        "--template",
        default=(
            "Rules:\n"
            "1) Use ONLY the passages below.\n"
            "2) Answer as a short span (1-6 words).\n"
            "3) If not answerable from the passages, output exactly UNKNOWN.\n"
            "4) Output format must be exactly two lines:\n"
            "ANSWER: <answer or UNKNOWN>\n"
            "EVIDENCE: <Px or NONE>\n\n"
            "Question: {question}\n\n"
            "Passages:\n{contexts}\n"
        ),
    )

    # Prompt-size controls (important when k is large, e.g., 200/500)
    ap.add_argument("--prompt-chunk-chars", type=int, default=1000)
    ap.add_argument("--max-context-chars", type=int, default=12000)
    ap.add_argument("--trace-chunk-chars", type=int, default=512)

    ap.add_argument("--out", required=True, help="Output JSONL run file")
    args = ap.parse_args()

    if args.limit is not None and args.max_questions is not None:
        raise SystemExit("Please specify only one of --limit or --max-questions.")

    items = load_answers(args.answers)
    n = args.limit if args.limit is not None else args.max_questions
    if n is not None:
        items = items[:n]

    index = load_tfidf_index(args.index)

    llm_cfg = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    if args.llm_backend == "ollama":
        llm = OllamaLLM(model=args.ollama_model, base_url=args.ollama_url)
    elif args.llm_backend == "openai":
        llm = OpenAILLM(model=args.openai_model)
    else:
        from rag_char_trace.llm.llama_cpp import LlamaCppLLM

        model_path = os.environ.get("LLAMA_CPP_MODEL_PATH")
        if not model_path:
            raise SystemExit(
                "For llama_cpp backend, set env var LLAMA_CPP_MODEL_PATH to a local .gguf model path."
            )
        llm = LlamaCppLLM(model_path=model_path)

    prompt = PromptSpec(system=args.system, template=args.template)

    w = JsonlTraceWriter(args.out)
    try:
        for it in tqdm(items, desc=f"RAG[{args.dataset}]", ncols=100):
            rec = run_one(
                item=it,
                index=index,
                llm=llm,
                llm_cfg=llm_cfg,
                prompt=prompt,
                top_k=args.k,
                prompt_max_hits=(None if args.k_use == 0 else max(0, args.k_use)),
                prompt_chunk_chars=args.prompt_chunk_chars,
                max_context_chars=args.max_context_chars,
                trace_chunk_chars=args.trace_chunk_chars,
            )
            w.write(rec)
    finally:
        w.close()

    print(f"Wrote run trace to: {args.out}")


if __name__ == "__main__":
    main()
