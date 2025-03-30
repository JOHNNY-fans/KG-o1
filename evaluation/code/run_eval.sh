#!/bin/bash

# vLLM Evaluation Service Launcher
# Description: This script starts a vLLM inference server and runs evaluation on specified datasets
# Usage: ./run_eval.sh

# Configuration Section
# ---------------------
CUDA_DEVICE=0
MODEL_PATH="MODEL_PATH"
SERVICE_PORT=8822
SERVED_MODEL_NAME="SERVED_MODEL_NAME"
MAX_RETRIES=10
RETRY_DELAY=5  # seconds

# --------------
echo "Starting vLLM inference service..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE vllm serve $MODEL_PATH \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port $SERVICE_PORT \
    --served-model-name $SERVED_MODEL_NAME \
    --gpu-memory-utilization 0.9 \
    --max_model_len 15000 \
    --enforce-eager &

VLLM_PID=$!
echo "vLLM service started with PID: $VLLM_PID"

# Run evaluation script
echo "Starting evaluation process..."
if [ -f "eval.py" ]; then
    python eval.py \
        --llm_name $SERVED_MODEL_NAME \
        --vllm_base_url "http://127.0.0.1:$SERVICE_PORT" \
        --dataset_type "hotpotqa,2wikimultihopqa,kg_mhqa,mintqa"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation script failed"
        exit 1
    fi
else
    echo "Error: eval.py script not found in current directory"
    exit 1
fi

echo "Evaluation completed successfully"
exit 0

