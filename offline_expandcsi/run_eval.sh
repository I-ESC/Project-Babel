#!/bin/bash

# Set the base path to the models and the script
BASE_PATH="alignment_runs_0624"
SCRIPT_PATH="main_linear_dai_order.py"

# List of model checkpoints with their respective paths
MODELS=(
    "${BASE_PATH}/offline_expand_offline_expandood_offline_expandcsi_offline_expandmmwave_MMFi_dataset/checkpoint_1_offline_expand.pt"
    "${BASE_PATH}/offline_expand_offline_expandood_offline_expandcsi_offline_expandmmwave_MMFi_dataset/checkpoint_2_offline_expandood.pt"
    "${BASE_PATH}/offline_expand_offline_expandood_offline_expandcsi_offline_expandmmwave_MMFi_dataset/checkpoint_3_offline_expandcsi.pt"
    "${BASE_PATH}/offline_expand_offline_expandood_offline_expandcsi_offline_expandmmwave_MMFi_dataset/checkpoint_4_offline_expandmmwave.pt"
    "${BASE_PATH}/offline_expand_offline_expandood_offline_expandcsi_offline_expandmmwave_MMFi_dataset/checkpoint_5_MMFi_dataset.pt"
    # "${BASE_PATH}/offline_expandood_MMFi_dataset_offline_expand_offline_expandcsi_offline_expandmmwave/checkpoint_1_offline_expandood.pt"
    # "${BASE_PATH}/offline_expandood_MMFi_dataset_offline_expand_offline_expandcsi_offline_expandmmwave/checkpoint_2_MMFi_dataset.pt"
    # "${BASE_PATH}/offline_expandood_MMFi_dataset_offline_expand_offline_expandcsi_offline_expandmmwave/checkpoint_3_offline_expand.pt"
    # "${BASE_PATH}/offline_expandood_MMFi_dataset_offline_expand_offline_expandcsi_offline_expandmmwave/checkpoint_4_offline_expandcsi.pt"
    # "${BASE_PATH}/offline_expandood_MMFi_dataset_offline_expand_offline_expandcsi_offline_expandmmwave/checkpoint_5_offline_expandmmwave.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_MMFi_dataset_offline_expand_offline_expandcsi/checkpoint_1_offline_expandood.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_MMFi_dataset_offline_expand_offline_expandcsi/checkpoint_2_offline_expandmmwave.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_MMFi_dataset_offline_expand_offline_expandcsi/checkpoint_3_MMFi_dataset.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_MMFi_dataset_offline_expand_offline_expandcsi/checkpoint_4_offline_expand.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_MMFi_dataset_offline_expand_offline_expandcsi/checkpoint_5_offline_expandcsi.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand/checkpoint_1_offline_expandood.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand/checkpoint_2_offline_expandmmwave.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand/checkpoint_3_offline_expandcsi.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand/checkpoint_4_MMFi_dataset.pt"
    # "${BASE_PATH}/offline_expandood_offline_expandmmwave_offline_expandcsi_MMFi_dataset_offline_expand/checkpoint_5_offline_expand.pt"
)

# Number of GPUs to use
NGPUS=4

# Function to run the script on a specific GPU
run_script() {
    local gpu=$1
    local model=$2
    echo "Running script with model ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 ${SCRIPT_PATH} --latest ${model} 
    echo "Finished running script with model ${model} on GPU ${gpu}"
}

# Iterate over each model and run the script on a different GPU
for ((i=0; i<${#MODELS[@]}; i++)); do
    gpu=$((i % NGPUS))
    run_script ${gpu} ${MODELS[$i]} &
done

# Wait for all background processes to finish
wait
