#!/bin/bash

cd ..
cd /workspace

git clone https://github.com/matthieudelsart/structured-outputs
pip install uv 
uv sync

uv huggingface-cli login --token XXX