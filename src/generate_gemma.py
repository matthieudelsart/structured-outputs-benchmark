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
from google import genai
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Any, List
from google.genai.types import GenerateContentConfigDict
import argparse

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv('GEMINI_KEY')

# Default model configuration
DEFAULT_MODEL = "gemma-3-1b-it"
GENERATION_CONFIG: GenerateContentConfigDict = {"temperature": 0.0}
DEFAULT_MAX_WORKERS = 2
GEMINI_MAX_WORKERS = 10

def get_max_workers(model_name: str) -> int:
    """
    Returns the appropriate number of workers based on the model name.
    
    Args:
        model_name: The name of the model being used.
        
    Returns:
        The number of workers to use for parallel processing.
    """
    return GEMINI_MAX_WORKERS if "gemini" in model_name.lower() else DEFAULT_MAX_WORKERS

def generate_with_retry(client: genai.client.Client, model_name: str, prompt: str, max_retries: int = 10, delay: int = 10) -> str | None:
    """
    Generates content using the Gemini API with a retry mechanism.

    This function sends a prompt to the Gemini model and will retry the request
    on failure, which is useful for handling intermittent network issues or API
    errors.

    Args:
        client: The initialized `google.genai.Client` instance.
        model_name: The name of the model to use.
        prompt: The complete prompt string to send to the model.
        max_retries: The maximum number of times to retry the API call.
        delay: The number of seconds to wait between retries.

    Returns:
        The generated text as a string, or None if the request fails after
        all retries.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GENERATION_CONFIG
            )
            return response.text
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Skipping this record.")
                return None


def process_batch(
    client: genai.client.Client,
    prompts: List[str],
    records: List[Dict[str, Any]],
    output_path: Path,
    task_name: str,
    model_name: str,
) -> None:
    """
    Processes a batch of prompts in parallel using a thread pool.

    Args:
        client: The initialized `google.genai.Client` instance.
        prompts: A list of complete prompt strings to be sent to the model.
        records: The corresponding list of original benchmark records.
        output_path: The file path where the final JSON output will be saved.
        task_name: The name of the task, used for the progress bar description.
        model_name: The name of the model to use for generation.
    """
    max_workers = get_max_workers(model_name)
    logger.info(f"Using {max_workers} workers for model {model_name}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # We use partial to pre-fill the `client` and `model_name` arguments of the generation function,
        # as `executor.map` only passes the item from the iterable (the prompt).
        bound_generate_func = partial(generate_with_retry, client, model_name)
        
        generated_texts = list(tqdm(
            executor.map(bound_generate_func, prompts),
            total=len(prompts),
            desc=f"Generating for {task_name}"
        ))

    results = []
    for record, generated_text in zip(records, generated_texts):
        new_record = record.copy()
        new_record['generated_output'] = generated_text
        results.append(new_record)

    with open(output_path / 'generated.json', 'w') as f:
        json.dump(results, f, indent=4)


def generate_wikibio(client: genai.client.Client, data_path: Path, output_path: Path, model_name: str, prompt_type: str = "") -> None:
    """
    Generates outputs for the `2-wiki_bio` task.

    This task requires dynamic prompt generation, as the list of expected JSON
    keys is specified uniquely for each record in the benchmark data.

    Args:
        client: The initialized `google.genai.Client` instance.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
        model_name: The name of the model to use for generation.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / f'prompt{"_" + prompt_type if prompt_type else ""}.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = []
    for record in bench_data:
        keys = ", ".join(f"'{k}'" for k in record['keys'])
        prompt = prompt_template.replace('{{EXPECTED_KEYS}}', keys)
        prompts.append(prompt + "\n\nInput: " + record['input'] + "\nOutput:")
        
    process_batch(client, prompts, bench_data, output_path, "2-wiki_bio", model_name)


def generate_apibank(client: genai.client.Client, data_path: Path, output_path: Path, model_name: str, prompt_type: str = "") -> None:
    """
    Generates outputs for the `5-api_bank` task.

    This task uses a prompt template that requires formatting with multiple
    fields from the benchmark record (`input` and `instruction`).

    Args:
        client: The initialized `google.genai.Client` instance.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
        model_name: The name of the model to use for generation.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / f'prompt{"_" + prompt_type if prompt_type else ""}.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = [
         (prompt_template
          .replace('{input}', record['input'])
          .replace('{instruction}', record['instruction']))
        for record in bench_data
    ]
    process_batch(client, prompts, bench_data, output_path, "5-api_bank", model_name)


def generate_generic(client: genai.client.Client, task_name: str, data_path: Path, output_path: Path, model_name: str, prompt_type: str = "") -> None:
    """
    A generic generator for tasks with a simple prompt structure.

    This function handles tasks where the prompt is constructed by either
    replacing an `{input}` placeholder or simply appending the input text.

    Args:
        client: The initialized `google.genai.Client` instance.
        task_name: The name of the task (e.g., "1-rotowire").
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
        model_name: The name of the model to use for generation.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / f'prompt{"_" + prompt_type if prompt_type else ""}.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

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
        
    process_batch(client, prompts, bench_data, output_path, task_name, model_name)


def main():
    """
    Main function to run the generation for all benchmark tasks.
    """
    parser = argparse.ArgumentParser(description="Generate benchmark outputs using the Gemini API.")
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        choices=["1-rotowire", "2-wiki_bio", "3-few_nerd", "4-TOPv1", "5-api_bank", "6-reasoning"],
        help="The task to start the generation from."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["1-rotowire", "2-wiki_bio", "3-few_nerd", "4-TOPv1", "5-api_bank", "6-reasoning"],
        help="Run only the specified task."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The name of the model to use (e.g., 'gemma-3-1b-it', 'gemma-7b-it')."
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="",
        choices=["low", "high"],
        help="The type of prompt to use (e.g., 'low', 'high')."
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f"generate_{args.model}{'_' + args.prompt_type if args.prompt_type else ''}.log"

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize the Gemini client
    client = genai.client.Client(api_key=GEMINI_KEY)

    # Define paths
    base_data_path = Path('data/clean')
    output_base_path = Path(f"results/{args.model}{'_' + args.prompt_type if args.prompt_type else ''}")

    tasks = {
        "1-rotowire": lambda c, d, o: generate_generic(c, "1-rotowire", d, o, args.model, args.prompt_type),
        "2-wiki_bio": lambda c, d, o: generate_wikibio(c, d, o, args.model, args.prompt_type),
        "3-few_nerd": lambda c, d, o: generate_generic(c, "3-few_nerd", d, o, args.model, args.prompt_type),
        "4-TOPv1": lambda c, d, o: generate_generic(c, "4-TOPv1", d, o, args.model, args.prompt_type),
        "5-api_bank": lambda c, d, o: generate_apibank(c, d, o, args.model, args.prompt_type),
    }

    # If a specific task is requested, run only that task
    if args.task:
        if args.task == "6-reasoning":
            logger.info("Processing task: 6-reasoning")
            reasoning_path = base_data_path / "6-reasoning"
            output_reasoning_path = output_base_path / "6-reasoning"
            for subtask in ["GSM8K", "last_letter"]:
                logger.info(f"Processing subtask: {subtask}")
                subtask_path = reasoning_path / subtask
                output_path = output_reasoning_path / subtask
                output_path.mkdir(parents=True, exist_ok=True)
                generate_generic(client, f"6-reasoning/{subtask}", subtask_path, output_path, args.model, args.prompt_type)
        elif args.task in tasks:
            logger.info(f"Processing task: {args.task}")
            data_path = base_data_path / args.task
            output_path = output_base_path / args.task
            output_path.mkdir(parents=True, exist_ok=True)
            tasks[args.task](client, data_path, output_path)
        else:
            logger.error(f"Invalid task name: {args.task}")
        return

    # Otherwise, run all tasks from the specified starting point
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
        gen_func(client, data_path, output_path)

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
            generate_generic(client, f"6-reasoning/{subtask}", subtask_path, output_path, args.model, args.prompt_type)


if __name__ == '__main__':
    main() 