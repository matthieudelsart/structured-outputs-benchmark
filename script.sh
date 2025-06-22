#bash script.sh >run.log 2>&1 

set -euo pipefail 

uv run python -m src.generate_transformers --model google/gemma-3-4b-it
uv run python -m src.generate_transformers_outlines --model google/gemma-3-4b-it
git add .
git commit -m "Add results transformers"
git push