"""
Generator script using HuggingFace *transformers* + Outlines JSON-schema
constrained decoding.

It mirrors the CLI and task logic of ``src.generate_vllm_outlines`` but
runs directly with transformers.  Unlike the vLLM version we do *not* need to
sanitize or strip advanced JSON-Schema keywords – Outlines can consume full
Draft-2019/2020 schemas.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from tqdm import tqdm
import outlines
from outlines.models.transformers import transformers as outlines_transformers

# Work around a bug in `datasets` pickling utilities where `log` may be missing
try:
    import datasets.utils._dill as _datasets_dill  # type: ignore

    if not hasattr(_datasets_dill, "log"):
        def _noop_log(*args, **kwargs):
            pass
        _datasets_dill.log = _noop_log  # type: ignore
except Exception:
    pass

# ───────────────────────────── Globals ────────────────────────────────────
TEMPERATURE = 0.0      # keep deterministic
MAX_NEW_TOKENS = 2048
MODEL_DOWNLOAD_DIR = Path("models"); MODEL_DOWNLOAD_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4         # Outlines is slower than vLLM – smaller batches
MAX_RECORDS: int | None = None

logger = logging.getLogger(__name__)

# Disable torch.compile globally (same as other script) ---------------------
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")
if hasattr(torch, "compile"):
    torch.compile = lambda m, *a, **kw: m  # type: ignore

# ───────────────────────── Helper: generation -----------------------------

from typing import Any


def _sanitize(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Outlines Draft 2020 helper – convert list-valued 'type' nodes to string.

    When the schema has constructs like {"type": ["integer", "null"]} the
    Outlines regex builder raises ``'type' must be a string``.  We keep the
    first non-"null" entry – good enough for benchmark evaluation which
    ignores nullability.
    """
    if isinstance(schema_dict, dict):
        for k, v in list(schema_dict.items()):
            if k == "type" and isinstance(v, list):
                new_v = [x for x in v if x != "null"]
                schema_dict[k] = new_v[0] if new_v else v[0]
            elif k in {"if", "then", "else", "oneOf", "anyOf", "allOf", "not", "propertyNames"} or k.startswith("$"):
                schema_dict.pop(k, None)
                continue
            else:
                _sanitize(v) if isinstance(v, dict) else None
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            _sanitize(item)
    return schema_dict


def _generate_batch(model: Any, prompts: List[str], schemas: List[Union[Dict[str, Any], str]], max_new_tokens: int | None = None) -> List[str]:
    """Generate a list of JSON outputs with constraints.

    Outlines does not support multi-prompt batching when constraints differ,
    so we loop serially inside the batch for now.  Still faster than fully
    serial outer loop because the model stays on device.
    """
    outs: List[str] = []
    for prompt, schema in zip(prompts, schemas):
        mnt = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS
        if isinstance(schema, (dict, list)):
            schema_fixed = _sanitize(json.loads(json.dumps(schema)))  # deep copy
            schema_obj = json.dumps(schema_fixed)
        else:
            schema_obj = schema
        gen_fn = outlines.generate.json(model, schema_obj)
        # Disable automatic JSON parsing to avoid runtime failures when model deviates
        gen_fn.format_sequence = lambda x: x  # type: ignore
        out = gen_fn(prompt, max_tokens=mnt)
        if isinstance(out, str):
            text = out.strip()
        else:
            # Convert python object (dict/list) to pretty JSON string
            text = json.dumps(out, ensure_ascii=False)
        outs.append(text)
    return outs

# ───────────────────────── Task helpers -----------------------------------


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text()


def _process_records(
    model: Any,
    prompts: List[str],
    schemas: List[Union[Dict[str, Any], str]],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
    *,
    max_new_tokens: int | None = None,
):
    output_path.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(prompts), desc=f"Generating ({task_name})")
    all_results: List[str] = []

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk_prompts = prompts[start : start + BATCH_SIZE]
        chunk_schemas = schemas[start : start + BATCH_SIZE]
        all_results.extend(_generate_batch(model, chunk_prompts, chunk_schemas, max_new_tokens))
        pbar.update(len(chunk_prompts))

    pbar.close()

    merged = [{**rec, "generated_output": gen} for rec, gen in zip(records, all_results)]
    with (output_path / "generated.json").open("w") as f:
        json.dump(merged, f, indent=4)


# generic task functions ----------------------------------------------------

def _generate_generic(
    model: Any,
    task_name: str,
    data_path: Path,
    output_path: Path,
    prompt_type: str = "",
    *,
    max_new_tokens: int | None = None,
):
    bench = json.load((data_path / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]

    prompt_tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")
    prompts: List[str] = []
    for rec in bench:
        prompt = (
            prompt_tpl.replace("{input}", rec["input"]) if "{input}" in prompt_tpl else prompt_tpl + f"\n\nInput: {rec['input']}\nOutput:"
        )
        prompts.append(prompt)

    # Determine schema(s)
    schema_file = data_path / "schema.json"
    if schema_file.exists():
        raw_schema = json.load(schema_file.open())
        schemas = [raw_schema] * len(prompts)
    else:
        schemas = [rec.get("json_schema", rec.get("schema")) for rec in bench]

    _process_records(model, prompts, schemas, bench, output_path, task_name, max_new_tokens=max_new_tokens)


def _generate_wikibio(model: Any, data_path: Path, output_path: Path, prompt_type: str = "", *, max_new_tokens: int | None = None):
    bench = json.load((data_path / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]
    tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")
    prompts, schemas = [], []
    for rec in bench:
        keys = ", ".join(f"'{k}'" for k in rec["keys"])
        prompt = tpl.replace("{{EXPECTED_KEYS}}", keys) + f"\n\nInput: {rec['input']}\nOutput:"
        prompts.append(prompt)
        schemas.append(rec.get("json_schema", rec.get("schema")))
    _process_records(model, prompts, schemas, bench, output_path, "2-wiki_bio", max_new_tokens=max_new_tokens)


def _generate_apibank(model: Any, data_path: Path, output_path: Path, prompt_type: str = "", *, max_new_tokens: int | None = None):
    bench = json.load((data_path / "bench.json").open())
    if MAX_RECORDS is not None:
        bench = bench[:MAX_RECORDS]
    tpl = _load_prompt(data_path / f"prompt{'_' + prompt_type if prompt_type else ''}.txt")
    prompts, schemas = [], []
    for rec in bench:
        prompt = tpl.replace("{input}", rec["input"]).replace("{instruction}", rec["instruction"])
        prompts.append(prompt)
        schemas.append(rec.get("json_schema", rec.get("schema")))
    _process_records(model, prompts, schemas, bench, output_path, "5-api_bank", max_new_tokens=max_new_tokens)

# ────────────────────────── Main ------------------------------------------

def main() -> None:
    global MAX_RECORDS, BATCH_SIZE, MAX_NEW_TOKENS

    p = argparse.ArgumentParser(description="Generate benchmark outputs using transformers + Outlines constrained decoding.")
    p.add_argument("--model", required=True, help="HF model name or path")
    p.add_argument("--prompt-type", choices=["", "low"], default="")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--max-records", type=int)
    p.add_argument("--start-from", choices=["1-rotowire","2-wiki_bio","3-few_nerd","4-TOPv1","5-api_bank","6-reasoning"])
    args = p.parse_args()

    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    MAX_RECORDS = args.max_records

    # Logging --------------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    model_slug = args.model.split("/")[-1]
    suff = f"_{args.prompt_type}" if args.prompt_type else ""
    log_path = logs_dir / f"generate_{model_slug}_transformers_outlines{suff}.log"
    prev = log_path.exists()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()], force=True)
    if prev:
        logger.info("\n===== RERUN invoked =====")

    # Load model via Outlines wrapper -------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = outlines_transformers(
        args.model,
        device=device,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "trust_remote_code": True,
        },
        tokenizer_kwargs={
            "cache_dir": str(MODEL_DOWNLOAD_DIR),
            "padding_side": "left",
        },
    )

    base = Path("data/clean")
    results_root = Path("results") / f"{model_slug}_transformers_outlines{suff}"

    tasks = {
        "1-rotowire": lambda: _generate_generic(model, "1-rotowire", base/"1-rotowire", results_root/"1-rotowire", args.prompt_type, max_new_tokens=args.max_new_tokens),
        "2-wiki_bio": lambda: _generate_wikibio(model, base/"2-wiki_bio", results_root/"2-wiki_bio", args.prompt_type, max_new_tokens=args.max_new_tokens),
        "3-few_nerd": lambda: _generate_generic(model, "3-few_nerd", base/"3-few_nerd", results_root/"3-few_nerd", args.prompt_type, max_new_tokens=args.max_new_tokens),
        "4-TOPv1": lambda: _generate_generic(model, "4-TOPv1", base/"4-TOPv1", results_root/"4-TOPv1", args.prompt_type, max_new_tokens=args.max_new_tokens),
        "5-api_bank": lambda: _generate_apibank(model, base/"5-api_bank", results_root/"5-api_bank", args.prompt_type, max_new_tokens=args.max_new_tokens),
    }

    if args.start_from and args.start_from != "6-reasoning":
        ordered = list(tasks.keys())
        start_idx = ordered.index(args.start_from)
        ordered = ordered[start_idx:]
    else:
        ordered = list(tasks.keys())

    for t in ordered:
        logger.info(f"Processing task: {t}")
        tasks[t]()

    if not args.start_from or args.start_from == "6-reasoning" or args.start_from not in tasks:
        for sub in ["GSM8K", "last_letter"]:
            name = f"6-reasoning/{sub}"
            logger.info(f"Processing task: {name}")
            _generate_generic(model, name, base/"6-reasoning"/sub, results_root/"6-reasoning"/sub, args.prompt_type, max_new_tokens=512)

if __name__ == "__main__":
    main() 