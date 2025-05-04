#!/bin/bash

# Set the path to the models and the script
MODEL_PATH="alignment_runs_0624/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand"
SCRIPT_PATH="main_linear_dai_order.py"

# List of model checkpoints
MODELS=(
    "checkpoint_1_offline_expandood.pt"
    "checkpoint_2_offline_expandmmwave.pt"
    "checkpoint_3_offline_expandcsi.pt"
    "checkpoint_4_MMFi_dataset.pt"
    "checkpoint_5_offline_expand.pt"
)

# Number of GPUs to use
NGPUS=4

# Function to run the script on a specific GPU
run_script() {
    local gpu=$1
    local model=$2
    echo "Running script with model ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 ${SCRIPT_PATH} --latest ${MODEL_PATH}/${model} 
    echo "Finished running script with model ${model} on GPU ${gpu}"
}

# Iterate over each model and run the script on a different GPU
for ((i=0; i<${#MODELS[@]}; i++)); do
    gpu=$((i % NGPUS))
    run_script ${gpu} ${MODELS[$i]} &
done

# Wait for all background processes to finish
wait
