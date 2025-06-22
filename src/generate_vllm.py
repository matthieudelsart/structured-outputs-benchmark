"""
Generator script using vLLM *without* constrained decoding.

This is a raw-generation variant of ``src.generate_vllm`` that keeps the same
CLI and data-handling logic so it can be dropped in existing automation (e.g.
``run_benchmarks.sh``) but omits all JSON-schema / guided-decoding features.

Differences from ``generate_vllm.py``:
• No Outlines integration, so no schema processing or guided decoding.
• Results and logs use the *_vllm* suffix instead of *_vllm*.

Supported benchmark tasks (identical to the constrained version):
1-rotowire, 2-wiki_bio, 3-few_nerd, 4-TOPv1, 5-api_bank, 6-reasoning (GSM8K &
last_letter).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore

# ─────────────────────────────── Globals ──────────────────────────────────────
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 2048
MODEL_DOWNLOAD_DIR = Path("models")  # where vLLM will cache HuggingFace models
MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Will be overridden by CLI (handy when debugging)
MAX_RECORDS: int | None = None

BATCH_SIZE = 16  # default, can be overridden by CLI

logger = logging.getLogger(__name__)

# ────────────────────────── vLLM core helpers ────────────────────────────────

def _process_records(
    llm: LLM,
    prompts: List[str],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
    *,
    max_new_tokens: int | None = None,
) -> None:
    """Generate completions for *prompts* in GPU batches using vLLM.

    This variant does *not* apply guided decoding – it uses plain sampling with
    deterministic settings so the behaviour is comparable to the constrained
    counterpart, minus the schema enforcement.
    """
    all_results: List[str] = []

    pbar = tqdm(total=len(prompts), desc=f"Generating ({task_name})")

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_new_tokens,
    )

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk_prompts = prompts[start : start + BATCH_SIZE]
        outputs = llm.generate(chunk_prompts, sampling_params)  # type: ignore[arg-type]
        for out in outputs:
            all_results.append(out.outputs[0].text.strip())
        pbar.update(len(chunk_prompts))

    pbar.close()

    # Merge with original records and save ------------------------------------------------
    merged: List[Dict[str, Any]] = []
    for rec, gen in zip(records, all_results):
        r = rec.copy()
        r["generated_output"] = gen
        merged.append(r)

    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "generated.json").open("w") as f:
        json.dump(merged, f, indent=4)


# ───────────────────────────── Task helpers ──────────────────────────────────

def _load_prompt(template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text()


def _generate_generic(
    llm: LLM,
    task_name: str,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
    *,
    max_new_tokens: int | None = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    bench = json.load((data_path / "bench.json").open())
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

    _process_records(
        llm,
        prompts,
        bench,
        output_path,
        task_name,
        max_new_tokens=max_new_tokens,
    )


def _generate_wikibio(
    llm: LLM,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    bench = json.load((data_path / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")

    prompts: List[str] = []
    for rec in bench:
        keys_fmt = ", ".join(f"'{k}'" for k in rec["keys"])
        prompt = prompt_tpl.replace("{{EXPECTED_KEYS}}", keys_fmt)
        prompt += f"\n\nInput: {rec['input']}\nOutput:"
        prompts.append(prompt)

    _process_records(
        llm,
        prompts,
        bench,
        output_path,
        "2-wiki_bio",
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _generate_apibank(
    llm: LLM,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    bench = json.load((data_path / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")
    prompts: List[str] = []
    for rec in bench:
        p = prompt_tpl.replace("{input}", rec["input"]).replace("{instruction}", rec["instruction"])
        prompts.append(p)

    _process_records(
        llm,
        prompts,
        bench,
        output_path,
        "5-api_bank",
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ────────────────────────────────  Main  ─────────────────────────────────────

def main() -> None:
    # Globals overridable by CLI ------------------------------------------------
    global MAX_RECORDS, BATCH_SIZE, MAX_NEW_TOKENS

    p = argparse.ArgumentParser(
        description="Generate benchmark outputs using vLLM (raw, no constrained decoding)."
    )
    p.add_argument("--model", type=str, required=True, help="HF model name or local path")
    p.add_argument(
        "--start-from",
        type=str,
        choices=[
            "1-rotowire",
            "2-wiki_bio",
            "3-few_nerd",
            "4-TOPv1",
            "5-api_bank",
            "6-reasoning",
        ],
        default=None,
        help="Start processing from this task (ignored if --task is given)",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=[
            "1-rotowire",
            "2-wiki_bio",
            "3-few_nerd",
            "4-TOPv1",
            "5-api_bank",
            "6-reasoning",
        ],
        default=None,
        help="Run only the specified task",
    )
    p.add_argument(
        "--prompt-type",
        type=str,
        choices=["", "low"],
        default="",
        help="Variant of the prompt to use (e.g. *_low.txt)",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the result directory name",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of records processed per task (debugging)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help=(
            "Maximum model length for vLLM. Increase if you encounter truncation issues."
        ),
    )

    # Performance tuning ---------------------------------------------------
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of prompts processed concurrently in each vLLM batch. Increase if you have spare VRAM to reduce request overhead.",
    )
    p.add_argument(
        "--tokenizer-pool-size",
        type=int,
        default=8,
        help="Parallel tokenization workers to accelerate request submission (0 = synchronous).",
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate for each completion.",
    )

    args = p.parse_args()
    MAX_RECORDS = args.max_records

    # Override globals based on CLI ---------------------------------------
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

    # Logging ----------------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    model_slug = args.model.split("/")[-1]
    prompt_suffix = f"_{args.prompt_type}" if args.prompt_type else ""
    results_root = Path("results") / f"{model_slug}_vllm{prompt_suffix}"
    log_path = logs_dir / f"generate_{model_slug}_vllm{prompt_suffix}.log"
    bench_time_path = Path("results") / "bench_results_time.json"
    previous_log_exists = log_path.exists()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Log rerun marker if appending to existing log ---------------------------
    if previous_log_exists:
        logger.info("\n===== RERUN invoked =====")

    # ─────────────────── Timers ──────────────────────
    task_times: Dict[str, float] = {}

    def _persist_times() -> None:
        """Safely write *task_times* (plus overall) to bench_results_time.json."""
        snapshot = {k: v for k, v in task_times.items()}
        snapshot["overall"] = sum(v for v in task_times.values())

        if bench_time_path.exists() and bench_time_path.stat().st_size > 0:
            try:
                data = json.loads(bench_time_path.read_text())
            except Exception:
                data = {}
        else:
            data = {}

        data[results_root.name] = snapshot
        with bench_time_path.open("w") as f:
            json.dump(data, f, indent=4)

    def _timeit(name: str, fn):
        logger.info(f"Processing task: {name}")
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        task_times[name] = elapsed
        logger.info(f"Task {name} completed in {elapsed:.2f} s")
        _persist_times()

    # Model ------------------------------------------------------------------
    logger.info(f"Loading model '{args.model}' with vLLM …")
    llm = LLM(
        model=args.model,
        download_dir=str(MODEL_DOWNLOAD_DIR.resolve()),
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tokenizer_pool_size=args.tokenizer_pool_size,
        tokenizer_pool_type="ray" if args.tokenizer_pool_size else None,
    )

    base_data = Path("data/clean")

    # Task dispatch ----------------------------------------------------------
    tasks = {
        "1-rotowire": lambda: _timeit(
            "1-rotowire",
            lambda: _generate_generic(
                llm, "1-rotowire", base_data / "1-rotowire", results_root / "1-rotowire", args.prompt_type
            ),
        ),
        "2-wiki_bio": lambda: _timeit(
            "2-wiki_bio",
            lambda: _generate_wikibio(
                llm, base_data / "2-wiki_bio", results_root / "2-wiki_bio", args.prompt_type
            ),
        ),
        "3-few_nerd": lambda: _timeit(
            "3-few_nerd",
            lambda: _generate_generic(
                llm, "3-few_nerd", base_data / "3-few_nerd", results_root / "3-few_nerd", args.prompt_type
            ),
        ),
        "4-TOPv1": lambda: _timeit(
            "4-TOPv1",
            lambda: _generate_generic(
                llm, "4-TOPv1", base_data / "4-TOPv1", results_root / "4-TOPv1", args.prompt_type
            ),
        ),
        "5-api_bank": lambda: _timeit(
            "5-api_bank",
            lambda: _generate_apibank(
                llm, base_data / "5-api_bank", results_root / "5-api_bank", args.prompt_type
            ),
        ),
    }

    def _run_reasoning() -> None:
        logger.info("Processing task: 6-reasoning")
        for sub in ["GSM8K", "last_letter"]:
            sub_name = f"6-reasoning/{sub}"
            logger.info(f"  > subtask {sub}")
            _timeit(
                sub_name,
                lambda: _generate_generic(
                    llm,
                    sub_name,
                    base_data / "6-reasoning" / sub,
                    results_root / "6-reasoning" / sub,
                    args.prompt_type,
                    max_new_tokens=512,
                ),
            )

    if args.task:
        if args.task == "6-reasoning":
            _run_reasoning()
        else:
            tasks[args.task]()
        return

    ordered = list(tasks.keys())

    # Handle start-from "6-reasoning" (skip numbered tasks) -----------------
    if args.start_from == "6-reasoning":
        start_idx = len(ordered)
    else:
        start_idx = ordered.index(args.start_from) if args.start_from else 0

    for t in ordered[start_idx:]:
        tasks[t]()

    # Run reasoning unless explicitly skipped --------------------------------
    if (
        args.start_from is None
        or args.start_from == "6-reasoning"
        or start_idx == 0
    ):
        _run_reasoning()


if __name__ == "__main__":
    main() 
    # uv run python -m src.generate_vllm --model google/gemma-3-4b-it