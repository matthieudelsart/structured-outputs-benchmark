#!/usr/bin/env python
"""Generate benchmark outputs using *Guidance* constrained decoding.

This script mirrors ``generate_vllm.py`` as closely as possible but replaces
vLLM + Outlines with the Guidance library (https://github.com/guidance-ai/guidance).
Guidance performs the JSON-schema-constrained decoding itself so we use a
regular HuggingFace Transformers model – this keeps the dependency surface
small and works on the same single-GPU A40 machine.

Due to Guidance's synchronous API we run the prompts sequentially. You can
speed things up by increasing ``--batch-size`` and enabling multi-processing
(see in-code note) but for typical benchmarking purposes the default settings
produce results in reasonable time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import guidance
from guidance import models as guidance_models
from tqdm import tqdm

# ───────────────────────────── Global config ────────────────────────────────

BATCH_SIZE = 1  # Guidance is synchronous; keep batches small
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024
MAX_RECORDS: int | None = None

logger = logging.getLogger("generate_guidance")

# ─────────────────────────── Helpers ─────────────────────────────────────────


def _load_prompt(template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text()


# Guidance program template ---------------------------------------------------
# We build a *single* compiled Guidance program for each unique prompt + schema
# and then reuse it. This is much faster than recompiling every iteration.

from functools import lru_cache


@lru_cache(maxsize=None)
def _compiled_program(prompt_template: str, schema_json: str) -> guidance.Program:  # type: ignore
    """Return a compiled Guidance program that asks the model to produce JSON
    conforming to *schema_json* after the prompt content."""

    program_text = (
        "{{#system}}You are a JSON-only assistant. For every request, you _must_ reply with **only** valid JSON, no prose.{{/system}}\n"
        + prompt_template
        + "\n\n{{#assistant}}{{json name='answer' schema=_schema}}{{/assistant}}"
    )
    return guidance(program_text)


# ─────────────────────────── Record processing ─────────────────────────────--


def _process_records(
    llm: guidance.LLM,  # type: ignore
    prompts: List[str],
    schemas: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
):
    output_path.mkdir(parents=True, exist_ok=True)

    merged: List[Dict[str, Any]] = []

    # Re-use a single TransformersSession across all records so that
    # Guidance can keep its prefix/KV cache ("token acceleration") hot.
    # This avoids re-encoding the shared system prompt + template for every
    # record and yields a noticeable speed-up on a single GPU.
    with llm.session() as session:
        for prompt, schema, rec in tqdm(
            zip(prompts, schemas, records),
            total=len(prompts),
            desc=f"Generating ({task_name})",
        ):
            schema_str = json.dumps(schema, sort_keys=True)
            program = _compiled_program(prompt, schema_str)
            # pass the *session* (not the bare llm) so the KV/cache sticks
            result = program(llm=session, _schema=json.loads(schema_str))
            gen_json = result["answer"].strip()
            merged.append({**rec, "generated_output": gen_json})

    with (output_path / "generated.json").open("w") as f:
        json.dump(merged, f, indent=4)


# ──────────────────────────── Task helpers ─────────────────────────────────--


def _load_bench(data_path: Path) -> List[Dict[str, Any]]:
    return json.load((data_path / "bench.json").open())


def _generate_generic(
    llm: guidance.LLM,  # type: ignore
    task_name: str,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
):
    bench = _load_bench(data_path)
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")

    prompts: List[str] = []
    for rec in bench:
        prompt = (
            prompt_tpl.replace("{input}", rec["input"])
            if "{input}" in prompt_tpl
            else prompt_tpl + f"\n\nInput: {rec['input']}\nOutput:"
        )
        prompts.append(prompt)

    schema_file = data_path / "schema.json"
    if schema_file.exists():
        schema = json.load(schema_file.open())
        schemas = [schema] * len(prompts)
    else:
        schemas = [rec["schema"] for rec in bench]

    _process_records(llm, prompts, schemas, bench, output_path, task_name)


def _generate_wikibio(
    llm: guidance.LLM,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
):
    bench = _load_bench(data_path)
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")

    prompts: List[str] = []
    for rec in bench:
        keys_fmt = ", ".join(f"'{k}'" for k in rec["keys"])
        prompt = prompt_tpl.replace("{{EXPECTED_KEYS}}", keys_fmt)
        prompt += f"\n\nInput: {rec['input']}\nOutput:"
        prompts.append(prompt)

    schemas = [rec["schema"] for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "2-wiki_bio")


def _generate_apibank(
    llm: guidance.LLM,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
):
    bench = _load_bench(data_path)
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")
    prompts = [prompt_tpl.format(input=rec["input"], instruction=rec["instruction"]) for rec in bench]
    schemas = [rec["schema"] for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "5-api_bank")


# ────────────────────────────────── main CLI ───────────────────────────────--


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark outputs using Guidance JSON constrained decoding.")
    p.add_argument("--model", required=True, type=str, help="HF model id or local path (e.g. google/gemma-3-4b-it)")
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--task", type=str, choices=[
        "1-rotowire",
        "2-wiki_bio",
        "3-few_nerd",
        "4-TOPv1",
        "5-api_bank",
        "6-reasoning",
    ], default=None, help="If given run only the specified task")
    p.add_argument("--prompt-type", type=str, choices=["", "low"], default="")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument(
    "--no-flash-attn",
    dest="flash_attn",
    action="store_false",
    help="Disable Flash-Attention-2 kernels (enabled by default)",
    )
    p.set_defaults(flash_attn=True)
    args = p.parse_args()

    global MAX_RECORDS
    MAX_RECORDS = args.max_records

    # ───────────────────────── Logging & paths ──────────────────────────
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)

    model_slug = args.model.split("/")[-1]
    prompt_suffix = f"_{args.prompt_type}" if args.prompt_type else ""

    results_root = Path("results") / f"{model_slug}_guidance{prompt_suffix}"
    log_path = logs_dir / f"generate_{model_slug}_guidance{prompt_suffix}.log"
    bench_time_path = Path("results") / "bench_results_time.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,  # override any previous configuration
    )

    # Duplicate stdout/stderr to the log file so that prints and tqdm bars are captured
    import sys, io

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._streams = streams

        def write(self, txt):
            for s in self._streams:
                s.write(txt)
                s.flush()
            return len(txt)

        def flush(self):
            for s in self._streams:
                s.flush()

    log_file_handle = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file_handle)
    sys.stderr = _Tee(sys.__stderr__, log_file_handle)

    # Load model via Guidance -------------------------------------------------
    logger.info(f"Loading model {args.model} via guidance.Transformers …")

    extra_model_kwargs = {}
    if args.flash_attn:
        # Newer versions of Transformers accept this kwarg and will fall back
        # gracefully if the model/config does not support Flash-Attention-2.
        extra_model_kwargs["attn_implementation"] = "flash_attention_2"

    llm = guidance_models.Transformers(
        args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        **extra_model_kwargs,
    )

    # ─────────────────── Timers ──────────────────────
    task_times: Dict[str, float] = {}

    base_data = Path("data/clean")

    def _persist_times() -> None:
        """Write *task_times* to bench_results_time.json incrementally."""
        # Compute overall on each write
        tmp = {k: v for k, v in task_times.items()}
        tmp["overall"] = sum(v for k, v in task_times.items())

        # Load existing file safe against emptiness/corruption
        if bench_time_path.exists() and bench_time_path.stat().st_size > 0:
            try:
                data = json.loads(bench_time_path.read_text())
            except Exception:
                data = {}
        else:
            data = {}

        data[results_root.name] = tmp
        with bench_time_path.open("w") as f:
            json.dump(data, f, indent=4)

    def _timeit(name: str, fn):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        task_times[name] = elapsed
        logger.info(f"Task {name} completed in {elapsed:.2f} s")
        _persist_times()

    tasks = {
        "1-rotowire": lambda: _timeit("1-rotowire", lambda: _generate_generic(llm, "1-rotowire", base_data / "1-rotowire", results_root / "1-rotowire", args.prompt_type)),
        "2-wiki_bio": lambda: _timeit("2-wiki_bio", lambda: _generate_wikibio(llm, base_data / "2-wiki_bio", results_root / "2-wiki_bio", args.prompt_type)),
        "3-few_nerd": lambda: _timeit("3-few_nerd", lambda: _generate_generic(llm, "3-few_nerd", base_data / "3-few_nerd", results_root / "3-few_nerd", args.prompt_type)),
        "4-TOPv1": lambda: _timeit("4-TOPv1", lambda: _generate_generic(llm, "4-TOPv1", base_data / "4-TOPv1", results_root / "4-TOPv1", args.prompt_type)),
        "5-api_bank": lambda: _timeit("5-api_bank", lambda: _generate_apibank(llm, base_data / "5-api_bank", results_root / "5-api_bank", args.prompt_type)),
    }

    def _run_reasoning() -> None:
        logger.info("Processing task: 6-reasoning")
        for sub in ["GSM8K", "last_letter"]:
            logger.info(f"  > subtask {sub}")
            sub_name = f"6-reasoning/{sub}"
            def _run_sub():
                _generate_generic(
                    llm,
                    sub_name,
                    base_data / "6-reasoning" / sub,
                    results_root / "6-reasoning" / sub,
                    args.prompt_type,
                )
            _timeit(sub_name, _run_sub)

    if args.task:
        if args.task == "6-reasoning":
            _run_reasoning()
        else:
            tasks[args.task]()
        return

    ordered = list(tasks.keys())
    for t in ordered:
        tasks[t]()

    # Run reasoning last
    _run_reasoning()


if __name__ == "__main__":
    main() 
    python -m src.generate_guidance --model google/gemma-3-4b-it 