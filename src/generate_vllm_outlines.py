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
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore
from vllm.model_executor.guided_decoding.outlines_logits_processors import build_regex_from_schema

# ─────────────────────────────── Globals ──────────────────────────────────────
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 2048
MODEL_DOWNLOAD_DIR = Path("models")  # where vLLM will cache HuggingFace models
MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Will be overridden by CLI (handy when debugging)
MAX_RECORDS: int | None = None

BATCH_SIZE = 16  # default, can be overridden by CLI

logger = logging.getLogger(__name__)

_SCHEMA_CACHE: Dict[str, str] = {}

# cache for regex strings per canonical JSON
_REGEX_CACHE: Dict[str, str] = {}

# Helper --------------------------------------------------------------------

def _regex_for_schema(schema: Dict[str, Any]) -> str:
    """Return a regex string representing *schema*, cached for reuse."""
    canon = json.dumps(schema, sort_keys=True)
    if canon not in _REGEX_CACHE:
        _REGEX_CACHE[canon] = build_regex_from_schema(canon, None)
    return _REGEX_CACHE[canon]

# Choose representation -----------------------------------------------------
def _guide_from_schema(schema: Dict[str, Any]) -> tuple[str, str]:
    """Return (mode, guide) where *mode* is either "json" or "regex"."""

    # If the schema defines several mutually-exclusive alternatives via
    # "oneOf", build a single regex that is the alternation (union) of the
    # per-branch regexes. This addresses API-Bank where each record provides
    # 4-5 possible call shapes.
    if isinstance(schema, dict) and "oneOf" in schema:
        parts: List[str] = []
        for alt in schema["oneOf"]:
            alt_regex = _regex_for_schema(_sanitize_schema(alt))
            # strip ^$ anchors so we can wrap once outside
            if alt_regex.startswith("^") and alt_regex.endswith("$"):
                alt_regex = alt_regex[1:-1]
            parts.append(f"(?:{alt_regex})")
        union_regex = "^" + "|".join(parts) + "$"
        return ("regex", union_regex)

    # Default: keep full (sanitised) JSON – richer semantic constraints.
    sanitized = _sanitize_schema(schema)
    # Preserve insertion order so that generation follows the schema order.
    # This matters for tasks like 6-reasoning where the expected JSON is
    # {"reasoning": …, "answer": …}.  Sorting alphabetically would flip the
    # keys and prompt the model to output the *answer* before its reasoning.
    return ("json", json.dumps(sanitized))

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
    schemas: List[Union[Dict[str, Any], str]],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
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

        def _to_gd_kwargs(guide_obj: Union[Dict[str, Any], str]):
            """Return kwargs for GuidedDecodingParams selecting *json* vs *regex*."""
            if isinstance(guide_obj, dict):
                return {"json": guide_obj}
            if isinstance(guide_obj, str) and _is_json_string(guide_obj):
                return {"json": guide_obj}
            # Fallback = regex string
            return {"regex": guide_obj}

        if all(s == chunk_schemas[0] for s in chunk_schemas):
            guide = chunk_schemas[0]
            gd_kwargs = _to_gd_kwargs(guide)
            shared_params = SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=max_new_tokens,
                guided_decoding=GuidedDecodingParams(backend="outlines", **gd_kwargs),
            )
            param_list = [shared_params] * len(chunk_prompts)
        else:
            param_list = [
                SamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=max_new_tokens,
                    guided_decoding=GuidedDecodingParams(backend="outlines", **_to_gd_kwargs(sc)),  # type: ignore[arg-type]
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
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
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
        raw_schema = json.load(schema_file.open())
        mode, guide = _guide_from_schema(raw_schema)
        schemas = [guide] * len(prompts)
    else:
        # For tasks that use per-record schema (wiki_bio & api_bank) we keep
        # them as-is. For all others we detect if they are identical and
        # canonicalise.
        if task_name in {"2-wiki_bio", "5-api_bank"}:
            schemas = [_sanitize_schema(rec.get("json_schema", rec.get("schema"))) for rec in bench]
        else:
            first = bench[0]["schema"]
            if all(rec["schema"] == first for rec in bench):
                _, guide = _guide_from_schema(first)
                schemas = [guide] * len(prompts)
            else:
                schemas = [
                    _sanitize_schema(rec.get("json_schema", rec.get("schema"))) if task_name not in {"2-wiki_bio", "5-api_bank"} else _sanitize_schema(rec.get("json_schema", rec.get("schema")))
                    for rec in bench
                ]

    _process_records(llm, prompts, schemas, bench, output_path, task_name, max_new_tokens=max_new_tokens)


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

    schemas = [_sanitize_schema(rec.get("json_schema", rec.get("schema"))) for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "2-wiki_bio", max_new_tokens=MAX_NEW_TOKENS)


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
    prompts = []
    for rec in bench:
        p = prompt_tpl.replace("{input}", rec["input"]).replace("{instruction}", rec["instruction"])
        prompts.append(p)
    schemas = [_sanitize_schema(rec.get("json_schema", rec.get("schema"))) for rec in bench]
    _process_records(llm, prompts, schemas, bench, output_path, "5-api_bank", max_new_tokens=MAX_NEW_TOKENS)


# ────────────────────── Schema sanitization for Outlines ────────────────

# Outlines' JSON-schema → regex builder currently supports a very restricted
# subset of Draft-07: essentially {type, properties, items, required, pattern,
# enum, additionalProperties, minItems/maxItems, minLength/maxLength}. Any
# other keyword triggers `ValueError`.
#
# The helper below converts an arbitrary (but local) schema to an Outlines-
# compatible one *without losing the structural information that matters for
# generation*:
#   • `$ref` to local `#/$defs/*` definitions are in-lined (recursively).
#   • keywords starting with `$` except `$ref` are dropped (meta only).
#   • conditional / boolean-logic keywords (`if`/`then`/`else`, `allOf`, …)
#     are dropped – Outlines can't express them in regex.
#   • `type` lists are collapsed to a single value (preferring "string").
#
# If the resulting dict has *none* of the supported structural keys we fall
# back to `{"type": "string"}` so that generation can continue.

_OUTLINES_KEYS = {
    "type",
    "properties",
    "items",
    "required",
    "pattern",
    "enum",
    "additionalProperties",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}


def _sanitize_schema(node: Any) -> Any:
    if isinstance(node, dict):
        # Resolve local definitions -------------------------------------------------
        local_defs = node.get("$defs", {})

        def _inline_ref(ref: str) -> Any:
            """Return the schema object referenced by a local $ref or a fallback."""
            if not (ref.startswith("#/$defs/") and len(ref.split("/")) >= 3):
                return {"type": "string"}
            name = ref.split("/", 2)[-1]
            target = local_defs.get(name)
            return _sanitize_schema(target) if target else {"type": "string"}

        new_obj: Dict[str, Any] = {}

        for k, v in node.items():
            # Drop meta keywords (start with $) that Outlines does not handle.
            if k.startswith("$") and k not in {"$ref"}:
                # Keep $defs for look-ups but do not copy into result.
                continue

            # Unsupported logical/conditional constructs --------------------------
            if k in {"if", "then", "else", "allOf", "anyOf", "oneOf", "not", "propertyNames"}:
                continue

            # Special handling for object properties --------------------------------
            if k == "properties" and isinstance(v, dict):
                new_obj[k] = {name: _sanitize_schema(sub) for name, sub in v.items()}
                continue

            # Inline $ref -----------------------------------------------------------
            if k == "$ref" and isinstance(v, str):
                inlined = _inline_ref(v)
                # Merge with whatever other keys were alongside $ref (rare)
                merged = {**inlined}
                for extra_k, extra_v in node.items():
                    if extra_k not in {"$ref", "$defs"}:
                        merged[extra_k] = _sanitize_schema(extra_v)
                return merged

            # Collapse list types --------------------------------------------------
            if k == "type" and isinstance(v, list):
                new_obj[k] = "string" if "string" in v else v[0]
            else:
                new_obj[k] = _sanitize_schema(v)

        # If nothing usable remains, default to string so Outlines still builds.
        if not any(key in _OUTLINES_KEYS for key in new_obj):
            return {"type": "string"}

        return new_obj

    if isinstance(node, list):
        return [_sanitize_schema(x) for x in node]

    return node


# ────────────────────────────────  Main  ─────────────────────────────────────

def main() -> None:
    # Globals overridable by CLI ------------------------------------------------
    global MAX_RECORDS, BATCH_SIZE, MAX_NEW_TOKENS

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
    results_root = Path("results") / f"{model_slug}_vllm_outlines{prompt_suffix}"
    log_path = logs_dir / f"generate_{model_slug}_vllm_outlines{prompt_suffix}.log"
    bench_time_path = Path("results") / "bench_results_time.json"
    previous_log_exists = log_path.exists()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,  # overwrite any prior basicConfig so vLLM logs propagate
    )

    # Log rerun marker (if we appended to an existing file) -------------------
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
        guided_decoding_backend="outlines",
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
                llm, "1-rotowire", base_data / "1-rotowire", results_root / "1-rotowire", args.prompt_type, max_new_tokens=MAX_NEW_TOKENS
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
                llm, "3-few_nerd", base_data / "3-few_nerd", results_root / "3-few_nerd", args.prompt_type, max_new_tokens=MAX_NEW_TOKENS
            ),
        ),
        "4-TOPv1": lambda: _timeit(
            "4-TOPv1",
            lambda: _generate_generic(
                llm, "4-TOPv1", base_data / "4-TOPv1", results_root / "4-TOPv1", args.prompt_type, max_new_tokens=MAX_NEW_TOKENS
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

    # Determine where to start in the 1-5 numbered tasks --------------------
    if args.start_from == "6-reasoning":
        # Skip the numbered tasks altogether – jump directly to reasoning.
        start_idx = len(ordered)
    else:
        start_idx = ordered.index(args.start_from) if args.start_from else 0

    for t in ordered[start_idx:]:
        tasks[t]()

    # Run reasoning if it wasn't explicitly skipped -------------------------
    if (
        args.start_from is None  # normal full run
        or args.start_from == "6-reasoning"  # user requested only reasoning
        or start_idx == 0  # we started from the very beginning
    ):
        _run_reasoning()


def _is_json_string(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main() 
    # uv run python -m src.generate_vllm_outlines --model google/gemma-3-4b-it 