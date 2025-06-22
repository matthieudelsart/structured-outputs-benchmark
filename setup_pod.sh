#!/bin/bash
python -m pip install --upgrade pip
git config --global user.email "matthieu.delsart@gmail.com"
git clone https://github.com/matthieudelsart/structured-outputs.git
cd structured-outputs
pip install uv 
uv sync
uv pip install flash-attn --no-build-isolation
uv run huggingface-cli login --token 