"""
Generator script using vLLM combined with the Outlines guided-decoding backend
so that every output strictly complies with a JSON schema.

Supported benchmark tasks
-------------------------
1-rotowire, 2-wiki_bio, 3-few_nerd, 4-TOPv1, 5-api_bank, 6-reasoning (GSM8K &
last_letter).

Some tasks rely on a single schema file (see *data/clean/<task>/schema.json*),
while others embed a dedicated ``schema`` field inside each record (wiki_bio &
api_bank). The logic mirrors what is implemented in *src.evaluate*.

The CLI stays almost identical to the previous generator so that existing
automation (e.g. *run_benchmarks.sh*) keeps working.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# ─────────────────────────────── Globals ──────────────────────────────────────
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024
MODEL_DOWNLOAD_DIR = Path("models")  # where vLLM will cache HuggingFace models
MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Will be overridden by CLI (handy when debugging)
MAX_RECORDS: int | None = None

BATCH_SIZE = 32  # default, can be overridden by CLI

logger = logging.getLogger(__name__)


# ────────────────────────── vLLM / Outlines core ─────────────────────────────

def _generate_json(llm: LLM, prompt: str, schema: Dict[str, Any]) -> str:
    """Generate a JSON string that satisfies *schema* using Outlines guidance."""
    guided = GuidedDecodingParams(json=schema, backend="outlines")
    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        guided_decoding=guided,
    )
    outputs = llm.generate(prompt, params)
    return outputs[0].outputs[0].text.strip()


def _process_records(
    llm: LLM,
    prompts: List[str],
    schemas: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
) -> None:
    """Generate outputs in GPU batches using vLLM's native batching.

    vLLM can accept *either* a single SamplingParams instance (applied to all
    prompts) *or* a list of SamplingParams (one per prompt). We build the latter
    so that every prompt can have its own JSON schema when required (wiki-bio,
    api_bank, reasoning, …).
    """
    all_results: List[str] = []

    pbar = tqdm(total=len(prompts), desc=f"Generating ({task_name})")

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk_prompts = prompts[start : start + BATCH_SIZE]
        chunk_schemas = schemas[start : start + BATCH_SIZE]

        param_list = [
            SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                guided_decoding=GuidedDecodingParams(json=sc, backend="outlines"),  # type: ignore
            )
            for sc in chunk_schemas
        ]

        # vLLM returns a list[RequestOutput]; .outputs[0].text is the first beam
        outputs = llm.generate(chunk_prompts, param_list)  # type: ignore[arg-type]
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

    # Determine schemas (static or dynamic)
    schema_file = data_path / "schema.json"
    if schema_file.exists():
        schema = json.load(schema_file.open())
        schemas = [_sanitize_schema(schema)] * len(prompts)
    else:
        schemas = [_sanitize_schema(rec["schema"]) for rec in bench]

    _process_records(llm, prompts, schemas, bench, output_path, task_name)


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

    schemas = [_sanitize_schema(rec["schema"]) for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "2-wiki_bio")


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
    prompts = [
        prompt_tpl.format(input=rec["input"], instruction=rec["instruction"])
        for rec in bench
    ]
    schemas = [_sanitize_schema(rec["schema"]) for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "5-api_bank")


# ────────────────────── Schema sanitization for Outlines ────────────────

def _sanitize_schema(node: Any) -> Any:
    """Return a copy of *node* where any JSON Schema `type` that is a list
    (e.g. ["string", "integer"]) is simplified to a single string that
    Outlines can handle (preferring "string" if present).

    Outlines' regex builder currently supports only string-valued types.  When
    it encounters a list, it raises `ValueError: 'type' must be a string`.
    The exact numeric/string distinction is already enforced in our schemas
    via the accompanying `pattern` regex, so casting to string is safe for the
    purposes of constrained generation.
    """

    if isinstance(node, dict):
        new_d: Dict[str, Any] = {}
        unsupported = {"if", "then", "else", "allOf", "anyOf", "oneOf", "not", "propertyNames", "$defs"}
        for k, v in node.items():
            if k in unsupported:
                # Drop these keys entirely – Outlines cannot handle them.
                continue
            if k == "type" and isinstance(v, list):
                # Prefer string; otherwise keep first.
                new_d[k] = "string" if "string" in v else v[0]
            else:
                new_d[k] = _sanitize_schema(v)
        return new_d
    if isinstance(node, list):
        return [_sanitize_schema(x) for x in node]
    return node


# ────────────────────────────────  Main  ─────────────────────────────────────

def main() -> None:
    # Globals overridable by CLI ------------------------------------------------
    global MAX_RECORDS, BATCH_SIZE

    p = argparse.ArgumentParser(
        description="Generate benchmark outputs using vLLM + Outlines constrained decoding."
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

    args = p.parse_args()
    MAX_RECORDS = args.max_records

    # Override globals based on CLI ---------------------------------------
    BATCH_SIZE = args.batch_size

    # Logging ----------------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"generate_{args.model.replace('/', '_')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )

    # Model ------------------------------------------------------------------
    logger.info(f"Loading model '{args.model}' with vLLM …")
    llm = LLM(
        model=args.model,
        download_dir=str(MODEL_DOWNLOAD_DIR.resolve()),
        trust_remote_code=True,
        guided_decoding_backend="outlines",
        max_model_len=args.max_model_len,
        tokenizer_pool_size=args.tokenizer_pool_size,
        tokenizer_pool_type="ray" if args.tokenizer_pool_size else None,
    )

    base_data = Path("data/clean")
    results_root = Path("results") / (
        f"{args.model.replace('/', '-')}{'_' + args.suffix if args.suffix else ''}{'_' + args.prompt_type if args.prompt_type else ''}"
    )

    # Task dispatch ----------------------------------------------------------
    tasks = {
        "1-rotowire": lambda: _generate_generic(
            llm, "1-rotowire", base_data / "1-rotowire", results_root / "1-rotowire", args.prompt_type
        ),
        "2-wiki_bio": lambda: _generate_wikibio(
            llm, base_data / "2-wiki_bio", results_root / "2-wiki_bio", args.prompt_type
        ),
        "3-few_nerd": lambda: _generate_generic(
            llm, "3-few_nerd", base_data / "3-few_nerd", results_root / "3-few_nerd", args.prompt_type
        ),
        "4-TOPv1": lambda: _generate_generic(
            llm, "4-TOPv1", base_data / "4-TOPv1", results_root / "4-TOPv1", args.prompt_type
        ),
        "5-api_bank": lambda: _generate_apibank(
            llm, base_data / "5-api_bank", results_root / "5-api_bank", args.prompt_type
        ),
    }

    def _run_reasoning() -> None:
        logger.info("Processing task: 6-reasoning")
        for sub in ["GSM8K", "last_letter"]:
            logger.info(f"  > subtask {sub}")
            _generate_generic(
                llm,
                f"6-reasoning/{sub}",
                base_data / "6-reasoning" / sub,
                results_root / "6-reasoning" / sub,
                args.prompt_type,
            )

    if args.task:
        if args.task == "6-reasoning":
            _run_reasoning()
        else:
            tasks[args.task]()
        return

    ordered = list(tasks.keys())
    start_idx = ordered.index(args.start_from) if args.start_from else 0
    for t in ordered[start_idx:]:
        tasks[t]()

    # Always run reasoning unless explicitly skipped by --start-from > last idx
    if not args.start_from or start_idx == 0:
        _run_reasoning()


if __name__ == "__main__":
    main() 