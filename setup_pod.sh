#!/bin/bash
python -m pip install --upgrade pip
pip install uv 
uv sync
git config --global user.email "matthieu.delsart@gmail.com"
uv pip install flash-attn --no-build-isolation
uv run huggingface-cli login --token xxx