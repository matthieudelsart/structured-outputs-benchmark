"""
This script generates JSON outputs for a series of benchmark tasks.

It automates the process of:
1.  Reading benchmark data and prompt templates for various tasks.
2.  Constructing specific prompts for each record in a benchmark.
3.  Calling the Google Gemini API to generate a JSON output based on the prompt.
4.  Running these API calls in parallel to accelerate the generation process.
5.  Saving the generated outputs alongside the original benchmark data.

The script is structured to handle the unique prompt-formatting requirements of each
benchmark task.
"""
# Standard libs
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time
from typing import Dict, Any, List
import argparse

# Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Model / generation parameters
# -----------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"
# Directory where the model will be downloaded (instead of HF default cache)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Generation config – keep temperature 0 for deterministic outputs
GENERATION_KWARGS = {
    "temperature": 0.0,  # ignored when do_sample False, kept for completeness
    "do_sample": False,
    "max_new_tokens": 256,
}

# Batch size – adjusted dynamically if you wish
BATCH_SIZE = 4

# Will be set from CLI; used by helpers to truncate benchmark
MAX_RECORDS: int | None = None

# Configure logging (handlers added in main)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    chats: List[List[Dict[str, str]]],
) -> List[str]:
    """Generate texts for a list of prompts in batches.

    Parameters
    ----------
    model / tokenizer : already loaded HF classes
    chats : list[list[dict[str, str]]]

    Returns
    -------
    list[str] – generated texts in the same order.
    """

    results: List[str] = []
    model.eval()  # type: ignore[attr-defined]

    with torch.no_grad():
        for i in range(0, len(chats), BATCH_SIZE):
            batch_chats = chats[i : i + BATCH_SIZE]

            batch_texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)  # type: ignore[attr-defined]
                for m in batch_chats
            ]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)  # type: ignore[attr-defined]

            output_ids = model.generate(**inputs, **GENERATION_KWARGS)  # type: ignore[attr-defined]

            # Remove the prompt part from generation and decode
            for j, ids in enumerate(output_ids):
                gen_ids = ids[inputs["input_ids"].shape[1] :]  # skip prompt tokens
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)  # type: ignore[arg-type]
                results.append(text.strip())

    return results

# New process_batch without threading (batch inference instead)

def process_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
) -> None:
    """Generate outputs for a list of prompts and save them.

    Uses batched inference instead of multi-threading, which is generally faster
    for local model execution.
    """

    generated_texts: List[str] = []

    pbar = tqdm(total=len(prompts), desc=f"Generating for {task_name}")
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_chats = [[{"role": "user", "content": p}] for p in batch_prompts]
        batch_texts = batch_generate(model, tokenizer, batch_chats)
        generated_texts.extend(batch_texts)
        pbar.update(len(batch_prompts))

    pbar.close()

    results = []
    for record, generated_text in zip(records, generated_texts):
        new_record = record.copy()
        new_record["generated_output"] = generated_text
        results.append(new_record)

    with open(output_path / "generated.json", "w") as f:
        json.dump(results, f, indent=4)

def generate_wikibio(model, tokenizer, data_path: Path, output_path: Path) -> None:
    """
    Generates outputs for the `2-wiki_bio` task.

    This task requires dynamic prompt generation, as the list of expected JSON
    keys is specified uniquely for each record in the benchmark data.

    Args:
        model: The loaded HF model.
        tokenizer: The corresponding HF tokenizer.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    if MAX_RECORDS is not None:
        bench_data = bench_data[:MAX_RECORDS]

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = []
    for record in bench_data:
        keys = ", ".join(f"'{k}'" for k in record['keys'])
        prompt = prompt_template.replace('{{EXPECTED_KEYS}}', keys)
        prompts.append(prompt + "\n\nInput: " + record['input'] + "\nOutput:")
        
    process_batch(model, tokenizer, prompts, bench_data, output_path, "2-wiki_bio")

def generate_apibank(model, tokenizer, data_path: Path, output_path: Path) -> None:
    """
    Generates outputs for the `5-api_bank` task.

    This task uses a prompt template that requires formatting with multiple
    fields from the benchmark record (`input` and `instruction`).

    Args:
        model: The loaded HF model.
        tokenizer: The corresponding HF tokenizer.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    if MAX_RECORDS is not None:
        bench_data = bench_data[:MAX_RECORDS]

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = [
        prompt_template.format(input=record['input'], instruction=record['instruction'])
        for record in bench_data
    ]
    process_batch(model, tokenizer, prompts, bench_data, output_path, "5-api_bank")

def generate_generic(model, tokenizer, task_name: str, data_path: Path, output_path: Path) -> None:
    """
    A generic generator for tasks with a simple prompt structure.

    This function handles tasks where the prompt is constructed by either
    replacing an `{input}` placeholder or simply appending the input text.

    Args:
        model: The loaded HF model.
        tokenizer: The corresponding HF tokenizer.
        task_name: The name of the task (e.g., "1-rotowire").
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    if MAX_RECORDS is not None:
        bench_data = bench_data[:MAX_RECORDS]

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = []
    for record in bench_data:
        full_prompt = prompt_template
        if '{input}' in prompt_template:
            full_prompt = prompt_template.replace('{input}', record['input'])
        else:
            full_prompt += "\n\nInput: " + record['input'] + "\nOutput:"
        prompts.append(full_prompt)
        
    process_batch(model, tokenizer, prompts, bench_data, output_path, task_name)

def main():
    """
    Main function to run the generation for all benchmark tasks.
    """
    parser = argparse.ArgumentParser(description="Generate benchmark outputs using the HF Qwen model.")
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        choices=[
            "1-rotowire",
            "2-wiki_bio",
            "3-few_nerd",
            "4-TOPv1",
            "5-api_bank",
            "6-reasoning",
        ],
        help="The task to start the generation from.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="If set, only generate for the first N records of each task (quick test run).",
    )

    args = parser.parse_args()

    # expose globally for helper functions
    global MAX_RECORDS
    MAX_RECORDS = args.max_records

    # ---------------- Logging ----------------
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f"generate_{MODEL_NAME.split('/')[-1]}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    # ---------------- Load HF model ----------------
    logger.info("Loading model %s to %s", MODEL_NAME, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_DIR, trust_remote_code=True
    )

    # Ensure correct padding side for decoder-only chat models
    tokenizer.padding_side = "left"

    # Detect best dtype / device
    use_mps = torch.backends.mps.is_available()
    dtype = torch.float16 if (torch.cuda.is_available() or use_mps) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",  # let Accelerate place weights (MPS / CUDA / CPU)
        trust_remote_code=True,
    )

    logger.info("Model loaded on device %s", model.device)

    # Define paths
    base_data_path = Path('data/clean')
    output_base_path = Path(f'results/{MODEL_NAME.split("/")[-1]}')

    tasks = {
        "1-rotowire": lambda m, t, d, o: generate_generic(m, t, "1-rotowire", d, o),
        "2-wiki_bio": generate_wikibio,
        "3-few_nerd": lambda m, t, d, o: generate_generic(m, t, "3-few_nerd", d, o),
        "4-TOPv1": lambda m, t, d, o: generate_generic(m, t, "4-TOPv1", d, o),
        "5-api_bank": generate_apibank,
    }

    task_names = list(tasks.keys())
    start_index = 0
    if args.start_from:
        try:
            start_index = task_names.index(args.start_from)
        except ValueError:
            logger.error(f"Invalid task name: {args.start_from}")
            return

    for i in range(start_index, len(task_names)):
        task_name = task_names[i]
        gen_func = tasks[task_name]
        logger.info(f"Processing task: {task_name}")
        data_path = base_data_path / task_name
        output_path = output_base_path / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        gen_func(model, tokenizer, data_path, output_path)

    # Special handling for 6-reasoning
    if not args.start_from or task_names.index(args.start_from) <= len(task_names):
        logger.info("Processing task: 6-reasoning")
        reasoning_path = base_data_path / "6-reasoning"
        output_reasoning_path = output_base_path / "6-reasoning"
        for subtask in ["GSM8K", "last_letter"]:
            logger.info(f"Processing subtask: {subtask}")
            subtask_path = reasoning_path / subtask
            output_path = output_reasoning_path / subtask
            output_path.mkdir(parents=True, exist_ok=True)
            generate_generic(model, tokenizer, f"6-reasoning/{subtask}", subtask_path, output_path)

if __name__ == '__main__':
    main() 

# uv run python -m src.generate_transformers --max-records 4 