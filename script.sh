#bash script.sh >run.log 2>&1 

set -euo pipefail 

# uv run python -m src.generate_transformers --model google/gemma-3-4b-it

# git add .
# git commit -m "Add results transformers"
# git push

uv run python -m src.generate_vllm --model google/gemma-3-12b-it >logs/run_vllm_12b.log 2>&1 
uv run python -m src.generate_vllm_outlines --model google/gemma-3-12b-it >logs/run_vllm_12b_outlines.log 2>&1 

git add .