#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo " NanoSpeech-MTL End-to-End Pipeline execution"
echo "============================================================"

# Create necessary directories
mkdir -p data/weights
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results

# ---------------------------------------------------------
# STEP 0: Data Acquisition (NanoListener & NanoBaseLib)
# ---------------------------------------------------------
echo "[0/4] Connecting to NanoBaseLib via NanoListener..."
# In a real environment, NanoListener would download the specific chunks here.
# Example: nanolistener fetch --dataset scBY4741_pU --out_dir data/raw
# For the pipeline execution, we assume the files are downloaded to data/raw/
FASTQ_FILE="data/raw/pass.fastq"
EVENTALIGN_FILE="data/raw/eventalign.txt"
RAW_TENSOR_FILE="data/raw/full_production_dataset.pt"

# Verify files exist before proceeding
if [ ! -f "$RAW_TENSOR_FILE" ]; then
    echo "Warning: Raw tensor dataset not found at $RAW_TENSOR_FILE."
    echo "Please ensure NanoListener has finished downloading NanoBaseLib chunks."
    exit 1
fi

# ---------------------------------------------------------
# STEP 1: Feature Extraction
# ---------------------------------------------------------
echo "\n[1/4] Extracting Tabular Biophysical Features..."
python src/extract_eventalign_features.py \
    --fastq $FASTQ_FILE \
    --eventalign $EVENTALIGN_FILE \
    --target_base T \
    --output data/processed/tabular_features.csv

# ---------------------------------------------------------
# STEP 2: Unsupervised Denoising
# ---------------------------------------------------------
echo "\n[2/4] Running GMM Physics Filter to Denoise Labels..."
python src/denoise_labels.py \
    --input $RAW_TENSOR_FILE \
    --mod_type pU \
    --output data/processed/clean_dataset.pt

# ---------------------------------------------------------
# STEP 3: Multi-Task CRNN Training
# ---------------------------------------------------------
echo "\n[3/4] Training the Multi-Task CRNN..."
python src/train_mtl.py \
    --dataset data/processed/clean_dataset.pt \
    --epochs 50 \
    --batch_size 1024 \
    --out_weights data/weights/nanospeech_mtl_best.pth

# ---------------------------------------------------------
# STEP 4: Live Inference Demo
# ---------------------------------------------------------
echo "\n[4/4] Running Live Inference Demo..."
python src/inference_demo.py \
    --dataset data/processed/clean_dataset.pt \
    --weights data/weights/nanospeech_mtl_best.pth \
    --num_samples 5000 \
    --output results/final_predictions.csv

echo "\n============================================================"
echo "Pipeline execution completed successfully!"
echo "Check the 'results/' folder for the final outputs."
echo "============================================================"
