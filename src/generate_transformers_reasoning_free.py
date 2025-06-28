#!/usr/bin/env python
"""Free-format generation for the *6-reasoning* benchmark subtasks (GSM8K, last_letter).

This is a lightweight wrapper around *transformers* that mirrors the loading,
logging, batching and CLI style of `src/generate_transformers.py`, but:

• **Only** the reasoning tasks are processed.
• Generation is *unconstrained* – the model is free to produce arbitrary text.
• A dedicated prompt instructs the model to first reason step-by-step and then
  output the final answer on a **new line starting with `Answer:`**.
• The full model output is preserved in a `full_answer` field, while the parsed
  actual answer (the portion after `Answer:`) is stored in the standard
  `generated_output` field so it can be evaluated like previous runs.

Usage example:

    python -m src.generate_transformers_reasoning_free \
        --model google/gemma-2b-it \
        --batch-size 8 --max-records 100
"""
from __future__ import annotations

import argparse, json, logging, os, re, time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Disable torch.compile globally (same safety belt as other scripts)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(torch, "compile"):
    torch.compile = lambda m, *a, **k: m  # type: ignore

# ─────────────────────────────── Globals ───────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16      # overridden by CLI
MAX_NEW_TOKENS = 512 # reasoning cap (same as vLLM scripts)
MAX_RECORDS: int | None = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: parse the *final* answer from a free-form completion --------------

def _parse_answer(text: str) -> str:
    """Return the substring following the *last* occurrence of "Answer:".

    If no explicit marker is found, fall back to the last non-empty line.
    """
    # Try regex capture (case-insensitive)
    matches = re.findall(r"Answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Fallback: last line
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return text.strip()

# ---------------------------------------------------------------------------
# Generation core -----------------------------------------------------------

def _process_batch(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompts: List[str],
    records: List[Dict[str, Any]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(prompts), desc="Generating (reasoning free)")

    # Ensure padding token exists so we can batch encode
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # helps Flash-Attention, if enabled

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk = prompts[start : start + BATCH_SIZE]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy deterministic decoding
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, out_ids in enumerate(gen_ids):
            inp_len = int(enc["attention_mask"][idx].sum())
            gen_tokens = out_ids[inp_len:]
            full = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            answer = _parse_answer(full)
            rec = records[start + idx].copy()
            rec["full_answer"] = full
            rec["generated_output"] = answer
            results.append(rec)
        pbar.update(len(chunk))

    pbar.close()

    with (output_dir / "generated.json").open("w") as f:
        json.dump(results, f, indent=4)

# ---------------------------------------------------------------------------
# Prompt helpers ------------------------------------------------------------

def _default_prompt(problem: str) -> str:
    """Fallback prompt instructing *chain-of-thought* followed by explicit answer."""
    return (
        "Solve the following problem step by step. After your reasoning, "
        "output the final answer on a new line in the exact format:\n"
        "Answer: <answer>\n\n"
        f"Problem: {problem}\n"
    )


def _load_prompt_file(p: Path) -> str | None:
    try:
        return p.read_text()
    except FileNotFoundError:
        return None

# ---------------------------------------------------------------------------
# Task wrappers -------------------------------------------------------------

def _run_subtask(
    model,
    tokenizer,
    subdir: Path,
    out_root: Path,
    prompt_file_name: str,
):
    bench = json.load((subdir / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_template = _load_prompt_file(subdir / prompt_file_name) or ""

    prompts: List[str] = []
    for rec in bench:
        if prompt_template:
            prompt = prompt_template.replace("{input}", rec["input"])
        else:
            prompt = _default_prompt(rec["input"])
        prompts.append(prompt)

    _process_batch(model, tokenizer, prompts, bench, out_root)

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------

def main() -> None:
    global MAX_RECORDS, BATCH_SIZE, MAX_NEW_TOKENS

    ap = argparse.ArgumentParser(description="Free-format reasoning generator (transformers)")
    ap.add_argument("--model", required=True, help="HF model name or path")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--prompt-file", default="prompt_reasoning.txt", help="Optional prompt file name inside each subtask directory")
    ap.add_argument("--flash-attn", action="store_true", help="Use FlashAttention-2 kernels (requires wheels)")

    args = ap.parse_args()
    BATCH_SIZE = args.batch_size
    MAX_RECORDS = args.max_records
    MAX_NEW_TOKENS = args.max_new_tokens

    # Logging setup --------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    slug = args.model.split("/")[-1]
    log_path = logs_dir / f"generate_{slug}_transformers_reasoning_free.log"
    prev = log_path.exists()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
                        force=True)
    if prev:
        logger.info("RERUN invoked")

    # Model & tokenizer ----------------------------------------------------
    if args.flash_attn:
        os.environ.setdefault("FLASH_ATTENTION_FORCE_UNPADDING", "1")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    model.eval()

    base = Path("data/clean/6-reasoning")
    out_root = Path("results") / f"{slug}_transformers_reasoning_free"

    task_times: Dict[str, float] = {}
    bench_time_path = Path("results") / "bench_results_time.json"

    def _timeit(name: str, fn):
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        task_times[name] = dt
        logger.info(f"Task {name} completed in {dt:.2f}s")
        # persist times ---------------------------------------------------
        snap = {**task_times, "overall": sum(task_times.values())}
        current = {}
        if bench_time_path.exists() and bench_time_path.stat().st_size > 0:
            try:
                current = json.loads(bench_time_path.read_text())
            except Exception:
                current = {}
        current[out_root.name] = snap
        bench_time_path.write_text(json.dumps(current, indent=4))

    # Run both subtasks ----------------------------------------------------
    subtasks = ["GSM8K", "last_letter"]
    for sub in subtasks:
        _timeit(sub, lambda s=sub: _run_subtask(
            model,
            tokenizer,
            base / s,
            out_root / s,
            args.prompt_file,
        ))

if __name__ == "__main__":
    main() 