"""
Generator script using HuggingFace *transformers* (no vLLM) for raw JSON
generation on the benchmark tasks.

It mirrors the CLI, logging, batching and per-task logic of
``src.generate_vllm_raw`` so that results are directly comparable.  The only
functional difference is the underlying inference backend.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import os

# Disable torch.compile globally before importing torch to avoid TorchDynamo traces
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# For older PyTorch versions, monkey-patch torch.compile to a no-op
if hasattr(torch, "compile"):
    def _no_compile(model, *args, **kwargs):
        return model  # just return the model unchanged

    torch.compile = _no_compile  # type: ignore[attr-defined]

# ─────────────────────────────── Globals ──────────────────────────────────────
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 2048
MODEL_DOWNLOAD_DIR = Path("models")
MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_RECORDS: int | None = None  # overridden by CLI
BATCH_SIZE = 16                 # overridden by CLI

logger = logging.getLogger(__name__)

# Detect device once -----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────── Core generation helper ───────────────────────────

def _process_records(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompts: List[str],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
    *,
    max_new_tokens: int | None = None,
) -> None:
    """Generate completions in GPU batches using *transformers*.

    For each prompt we decode only the *generated* tokens (the portion after
    the prompt) so that the output format matches what vLLM returns.
    """
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    all_results: List[str] = []
    pbar = tqdm(total=len(prompts), desc=f"Generating ({task_name})")

    # Ensure padding token exists (needed for batching)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk_prompts = prompts[start : start + BATCH_SIZE]

        enc = tokenizer(
            chunk_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy deterministic
                pad_token_id=tokenizer.pad_token_id,
            )

        # Split batch outputs -------------------------------------------------
        for idx, output_ids in enumerate(gen_ids):
            input_len = int(enc["attention_mask"][idx].sum())
            gen_tokens = output_ids[input_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_results.append(text)

        pbar.update(len(chunk_prompts))

    pbar.close()

    # Merge & persist ---------------------------------------------------------
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
    model,
    tokenizer,
    task_name: str,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
    *,
    max_new_tokens: int | None = None,
):
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
        model,
        tokenizer,
        prompts,
        bench,
        output_path,
        task_name,
        max_new_tokens=max_new_tokens,
    )


def _generate_wikibio(
    model,
    tokenizer,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
):
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
        model,
        tokenizer,
        prompts,
        bench,
        output_path,
        "2-wiki_bio",
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _generate_apibank(
    model,
    tokenizer,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
):
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
        model,
        tokenizer,
        prompts,
        bench,
        output_path,
        "5-api_bank",
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ────────────────────────────────  Main  ─────────────────────────────────────

def main() -> None:
    global MAX_RECORDS, BATCH_SIZE, MAX_NEW_TOKENS

    p = argparse.ArgumentParser(description="Generate benchmark outputs using HuggingFace transformers (raw).")
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
    p.add_argument("--prompt-type", type=str, choices=["", "low"], default="", help="Variant of the prompt to use (e.g. *_low.txt)")
    p.add_argument("--suffix", type=str, default=None, help="Optional suffix appended to the result directory name")
    p.add_argument("--max-records", type=int, default=None, help="Limit the number of records processed per task (debugging)")
    p.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length (ignored by some models)")

    # Perf-tuning
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Prompts per generation batch")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum tokens to generate per completion")

    # FlashAttention-2 opt-in -------------------------------------------------
    p.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use FlashAttention-2 kernels (requires flash-attn wheels)",
    )


    args = p.parse_args()
    MAX_RECORDS = args.max_records
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

    # Logging ---------------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    model_slug = args.model.split("/")[-1]
    prompt_suffix = f"_{args.prompt_type}" if args.prompt_type else ""
    results_root = Path("results") / f"{model_slug}_transformers{prompt_suffix}"
    log_path = logs_dir / f"generate_{model_slug}_transformers{prompt_suffix}.log"
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

    if previous_log_exists:
        logger.info("\n===== RERUN invoked =====")

    # Model -----------------------------------------------------------------
    logger.info(f"Loading model '{args.model}' with transformers on {DEVICE} …")

    # Flash-attention env hint
    if args.flash_attn:
        os.environ.setdefault("FLASH_ATTENTION_FORCE_UNPADDING", "1")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(MODEL_DOWNLOAD_DIR))

    # Choose attention impl
    attn_impl = "flash_attention_2" if args.flash_attn else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        cache_dir=str(MODEL_DOWNLOAD_DIR),
        attn_implementation=attn_impl,
    )
    # Disable torch_compile at the config level to avoid Dynamo issues
    for cfg in (getattr(model, 'generation_config', None), getattr(model, 'config', None)):
        if cfg is not None and hasattr(cfg, 'torch_compile'):
            cfg.torch_compile = False
    model.eval()

    base_data = Path("data/clean")

    # ─── Timers ───────────────────────────────────────────────────────────
    task_times: Dict[str, float] = {}

    def _persist_times() -> None:
        snap = {k: v for k, v in task_times.items()}
        snap["overall"] = sum(task_times.values())
        if bench_time_path.exists() and bench_time_path.stat().st_size > 0:
            try:
                data = json.loads(bench_time_path.read_text())
            except Exception:
                data = {}
        else:
            data = {}
        data[results_root.name] = snap
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

    # Task dispatch ---------------------------------------------------------
    tasks = {
        "1-rotowire": lambda: _timeit(
            "1-rotowire",
            lambda: _generate_generic(
                model,
                tokenizer,
                "1-rotowire",
                base_data / "1-rotowire",
                results_root / "1-rotowire",
                args.prompt_type,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
        ),
        "2-wiki_bio": lambda: _timeit(
            "2-wiki_bio",
            lambda: _generate_wikibio(
                model,
                tokenizer,
                base_data / "2-wiki_bio",
                results_root / "2-wiki_bio",
                args.prompt_type,
            ),
        ),
        "3-few_nerd": lambda: _timeit(
            "3-few_nerd",
            lambda: _generate_generic(
                model,
                tokenizer,
                "3-few_nerd",
                base_data / "3-few_nerd",
                results_root / "3-few_nerd",
                args.prompt_type,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
        ),
        "4-TOPv1": lambda: _timeit(
            "4-TOPv1",
            lambda: _generate_generic(
                model,
                tokenizer,
                "4-TOPv1",
                base_data / "4-TOPv1",
                results_root / "4-TOPv1",
                args.prompt_type,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
        ),
        "5-api_bank": lambda: _timeit(
            "5-api_bank",
            lambda: _generate_apibank(
                model,
                tokenizer,
                base_data / "5-api_bank",
                results_root / "5-api_bank",
                args.prompt_type,
            ),
        ),
    }

    def _run_reasoning():
        logger.info("Processing task: 6-reasoning")
        for sub in ["GSM8K", "last_letter"]:
            sub_name = f"6-reasoning/{sub}"
            logger.info(f"  > subtask {sub}")
            _timeit(
                sub_name,
                lambda: _generate_generic(
                    model,
                    tokenizer,
                    sub_name,
                    base_data / "6-reasoning" / sub,
                    results_root / "6-reasoning" / sub,
                    args.prompt_type,
                    max_new_tokens=512,
                ),
            )

    # Select tasks ----------------------------------------------------------
    if args.task:
        if args.task == "6-reasoning":
            _run_reasoning()
        else:
            tasks[args.task]()
        return

    ordered = list(tasks.keys())
    if args.start_from == "6-reasoning":
        start_idx = len(ordered)
    else:
        start_idx = ordered.index(args.start_from) if args.start_from else 0

    for t in ordered[start_idx:]:
        tasks[t]()

    if (
        args.start_from is None
        or args.start_from == "6-reasoning"
        or start_idx == 0
    ):
        _run_reasoning()


if __name__ == "__main__":
    main() 