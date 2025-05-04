#!/bin/bash

# Base directory for saving models and logs
BASE_DIR="alignment_runs_0501"
mkdir -p $BASE_DIR

# Dataset orders with corresponding directories
ORDERS=(
    # "offline_expand offline_expandood offline_expandcsi offline_expandmmwave MMFi_dataset"
    # "offline_expandood MMFi_dataset offline_expand offline_expandcsi offline_expandmmwave"
    # "offline_expandood offline_expandmmwave offline_expandcsi MMFi_dataset offline_expand"
    "offline_expandood offline_expandmmwave MMFi_dataset offline_expand offline_expandcsi"
)

# Number of GPUs available
NUM_GPUS=4

# Iterate through each order
for ((i=0; i<${#ORDERS[@]}; i++)); do
    ORDER=${ORDERS[$i]}
    # GPU_ID=1
    GPU_ID=$((i % NUM_GPUS))

    # Generate a unique directory name for this order
    ORDER_DIR=$(echo $ORDER | tr ' ' '_')
    OUTPUT_DIR="$BASE_DIR/$ORDER_DIR"
    mkdir -p $OUTPUT_DIR

    # Get the parent directory of OUTPUT_DIR
    PARENT_DIR=$(dirname "$OUTPUT_DIR")

    # Split the ORDER into an array
    IFS=' ' read -r -a DATASETS <<< "$ORDER"

    # Initialize the checkpoint filename for loading
    INITIAL_CHECKPOINT=""

    # Iterate through each dataset in the order with enumeration
    for ((j=0; j<${#DATASETS[@]}; j++)); do
        DATASET=${DATASETS[$j]}
        STEP=$((j + 1))

        # Generate the checkpoint filenames based on the current dataset and order
        LOAD_FILENAME="$INITIAL_CHECKPOINT"
        SAVE_FILENAME="checkpoint_${STEP}_${DATASET}.pt"

        # Check if the checkpoint file already exists
        if [ -f "$OUTPUT_DIR/$SAVE_FILENAME" ]; then
            echo "Checkpoint for dataset $DATASET (Step: $STEP) already exists, skipping..."
            INITIAL_CHECKPOINT="$OUTPUT_DIR/$SAVE_FILENAME"
            continue
        fi

        # Change to the dataset directory
        cd $DATASET

        # Run the alignment script on the specified GPU
        echo "Running alignment for dataset: $DATASET (Step: $STEP) in order: $ORDER on GPU $GPU_ID"
        if [ -z "$LOAD_FILENAME" ]; then
            # First run doesn't have a checkpoint to load
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_dai_order.py \
                --output-dir $OUTPUT_DIR \
                --epochs 500 \
                --batch-size 256 \
                --lr 1e-4 \
                --wd 1e-4 \
                --save-filename $SAVE_FILENAME \
                --version 0 \
                --seed 0
        else
            # Subsequent runs load the previous checkpoint
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_dai_order.py \
                --output-dir $OUTPUT_DIR \
                --epochs 500 \
                --batch-size 256 \
                --lr 1e-4 \
                --wd 1e-4 \
                --load-filename $LOAD_FILENAME \
                --save-filename $SAVE_FILENAME \
                --version 0 \
                --seed 0
        fi

        # Change back to the base directory
        cd -

        # Update the checkpoint filename for the next dataset in the order
        INITIAL_CHECKPOINT="$OUTPUT_DIR/$SAVE_FILENAME"
    done &
done

# Wait for all background jobs to finish
wait

echo "All alignments completed."
