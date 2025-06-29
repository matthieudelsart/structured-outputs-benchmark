#bash script.sh >run.log 2>&1 

set -euo pipefail 

# uv run python -m src.generate_transformers --model google/gemma-3-4b-it

# export HUGGING_FACE_HUB_TOKEN=XX


# uv run python -m src.generate_vllm --model google/gemma-3-1b-it >logs/run_vllm_1b.log 2>&1 
uv run python -m src.generate_vllm_outlines --model google/gemma-3-1b-it --start-from 3-few_nerd >>logs/run_vllm_1b_outlines.log 2>&1 

git add .