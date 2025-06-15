#!/bin/bash

# List of models to run
MODELS=(
    "gemma-3-27b-it"
    "gemma-3-4b-it"
    "gemma-3-1b-it"
   # "gemma-3-12b-it"
    "gemini-2.5-flash-preview-05-20"

)

# List of tasks to run for each model
TASKS=(
    "5-api_bank"
)

# Function to run a single model-task combination
run_model_task() {
    local model=$1
    local task=$2
    
    # Skip 1-rotowire task for gemma-3-1b-it model
    if [ "$task" = "4-TOPv1" ] && [ "$model" != "gemma-3-12b-it" ]; then
        echo "Skipping task $task for model $model (not supported)"
        return
    fi
    
    echo "Running model $model on task $task"
    python -m src.generate_gemma --model "$model" --task "$task"
}

# Main loop
for model in "${MODELS[@]}"; do
    echo "Starting runs for model: $model"
    for task in "${TASKS[@]}"; do
        run_model_task "$model" "$task"
    done
    echo "Completed all tasks for model: $model"
done

echo "All benchmarks completed!"  

#uv run ./run_benchmarks.sh & pid=$!  
# caffeinate -i -w $pid

#uv run python -m src.generate_gemma --model gemma-3-12b-it & pid=$!  
#caffeinate -i -w $pid