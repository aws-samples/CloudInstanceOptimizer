#!/bin/bash

PORT=29500
NGPU=$(command -v nvidia-smi &> /dev/null && nvidia-smi --list-gpus | wc -l || echo 0)
EXPECTED_NODES="${AWS_BATCH_JOB_NUM_NODES:-1}"
NODE_RANK="${AWS_BATCH_JOB_NODE_INDEX:-0}"

# Get main node IP from AWS Batch environment variable
MAIN_NODE_IP="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-localhost}"
RDZV_ENDPOINT="${MAIN_NODE_IP}:$PORT"

# Generate a unique ID from the AWS Batch job ID
JOB_ID="${AWS_BATCH_JOB_ID:-$(date +%s)}"

echo "===== DISTRIBUTED TRAINING CONFIGURATION ====="
echo "Number of GPUs per node: $NGPU"
echo "Total number of nodes: $EXPECTED_NODES"
echo "Current node rank: $NODE_RANK"
echo "Main node IP: $MAIN_NODE_IP"
echo "Rendezvous endpoint: $RDZV_ENDPOINT"
echo "Job ID: $JOB_ID"
echo "==========================================="

# Wait briefly on worker nodes to ensure the main node is ready
if [ "$NODE_RANK" -eq "0" ]; then
    echo "Head node waiting 60 seconds for worker nodes to be ready..."
    sleep 60
fi

echo "Starting pytorch"

# Launch PyTorch distributed training
python -m torch.distributed.run \
    --nproc_per_node=$NGPU \
    --nnodes=$EXPECTED_NODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    train.py

