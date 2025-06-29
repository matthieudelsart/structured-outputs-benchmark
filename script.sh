#bash script.sh >run.log 2>&1 

set -euo pipefail 

# uv run python -m src.generate_transformers --model google/gemma-3-4b-it

# export HUGGING_FACE_HUB_TOKEN=XX


uv run python -m src.generate_transformers_outlines --model google/gemma-3-4b-it --start-from 6-reasoning >>logs/run_transformers_4b_outlines.log 2>&1 


git add .