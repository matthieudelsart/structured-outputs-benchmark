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

# Will be overridden by CLI (handy when debugging)
MAX_RECORDS: int | None = None

BATCH_SIZE = 32  # tune depending on memory

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

    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Generating ({task_name})"):
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
        schemas = [schema] * len(prompts)
    else:
        schemas = [rec["schema"] for rec in bench]

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

    schemas = [rec["schema"] for rec in bench]
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
    schemas = [rec["schema"] for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "5-api_bank")


# ────────────────────────────────  Main  ─────────────────────────────────────

def main() -> None:
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

    args = p.parse_args()
    global MAX_RECORDS
    MAX_RECORDS = args.max_records

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
        dtype="float16",
        download_dir=str(MODEL_DOWNLOAD_DIR),
        trust_remote_code=True,
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