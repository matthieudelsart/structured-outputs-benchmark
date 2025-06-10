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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv('GEMINI_KEY')

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GENERATION_CONFIG: GenerateContentConfigDict = {"temperature": 0.0}
MAX_WORKERS = 10

def generate_with_retry(client: genai.client.Client, prompt: str, max_retries: int = 5, delay: int = 5) -> str | None:
    """
    Generates content using the Gemini API with a retry mechanism.

    This function sends a prompt to the Gemini model and will retry the request
    on failure, which is useful for handling intermittent network issues or API
    errors.

    Args:
        client: The initialized `google.genai.Client` instance.
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
                model=MODEL_NAME,
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
) -> None:
    """
    Processes a batch of prompts in parallel using a thread pool.

    Args:
        client: The initialized `google.genai.Client` instance.
        prompts: A list of complete prompt strings to be sent to the model.
        records: The corresponding list of original benchmark records.
        output_path: The file path where the final JSON output will be saved.
        task_name: The name of the task, used for the progress bar description.
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # We use partial to pre-fill the `client` argument of the generation function,
        # as `executor.map` only passes the item from the iterable (the prompt).
        bound_generate_func = partial(generate_with_retry, client)
        
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


def generate_wikibio(client: genai.client.Client, data_path: Path, output_path: Path) -> None:
    """
    Generates outputs for the `2-wiki_bio` task.

    This task requires dynamic prompt generation, as the list of expected JSON
    keys is specified uniquely for each record in the benchmark data.

    Args:
        client: The initialized `google.genai.Client` instance.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = []
    for record in bench_data:
        keys = ", ".join(f"'{k}'" for k in record['keys'])
        prompt = prompt_template.replace('{{EXPECTED_KEYS}}', keys)
        prompts.append(prompt + "\n\nInput: " + record['input'] + "\nOutput:")
        
    process_batch(client, prompts, bench_data, output_path, "2-wiki_bio")


def generate_apibank(client: genai.client.Client, data_path: Path, output_path: Path) -> None:
    """
    Generates outputs for the `5-api_bank` task.

    This task uses a prompt template that requires formatting with multiple
    fields from the benchmark record (`input` and `instruction`).

    Args:
        client: The initialized `google.genai.Client` instance.
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

    with open(bench_file, 'r') as f:
        bench_data = json.load(f)

    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    prompts = [
        prompt_template.format(input=record['input'], instruction=record['instruction'])
        for record in bench_data
    ]
    process_batch(client, prompts, bench_data, output_path, "5-api_bank")


def generate_generic(client: genai.client.Client, task_name: str, data_path: Path, output_path: Path) -> None:
    """
    A generic generator for tasks with a simple prompt structure.

    This function handles tasks where the prompt is constructed by either
    replacing an `{input}` placeholder or simply appending the input text.

    Args:
        client: The initialized `google.genai.Client` instance.
        task_name: The name of the task (e.g., "1-rotowire").
        data_path: The path to the directory containing the task's data.
        output_path: The path to the directory where results will be saved.
    """
    bench_file = data_path / 'bench.json'
    prompt_file = data_path / 'prompt.txt'

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
        
    process_batch(client, prompts, bench_data, output_path, task_name)


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
    args = parser.parse_args()

    client = genai.Client(api_key=GEMINI_KEY)

    # Define paths
    base_data_path = Path('data/clean')
    output_base_path = Path('data/generated')

    tasks = {
        "1-rotowire": lambda c, d, o: generate_generic(c, "1-rotowire", d, o),
        "2-wiki_bio": generate_wikibio,
        "3-few_nerd": lambda c, d, o: generate_generic(c, "3-few_nerd", d, o),
        "4-TOPv1": lambda c, d, o: generate_generic(c, "4-TOPv1", d, o),
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
            generate_generic(client, f"6-reasoning/{subtask}", subtask_path, output_path)


if __name__ == '__main__':
    main() 