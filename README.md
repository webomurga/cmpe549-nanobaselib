# NanoSpeech-MTL: Biophysical RNA Modification Detector

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![NanoBaseLib](https://img.shields.io/badge/Dataset-NanoBaseLib-success.svg)

This repository contains the complete Multi-Task Learning (MTL) pipeline for detecting RNA modifications (like Pseudouridine and m6A) directly from raw Nanopore electrical signals.

Unlike standard basecallers, this pipeline mathematically purifies the dataset using unsupervised biophysics (Gaussian Mixture Models) to overcome stoichiometric label noise before training a dual-headed CRNN to simultaneously basecall and detect modifications.

Standard deep learning sequence models often struggle on raw Nanopore data due to **Stoichiometric Label Noise**-cells do not modify 100% of their RNA, meaning human-provided ground-truth datasets are filled with contradictory labels. 

NanoSpeech-MTL solves this by abandoning human labels and utilizing **Biophysics**:
1. **Unsupervised Denoising:** A Gaussian Mixture Model (GMM) analyzes the raw electrical physics to discover true modifications. It hunts for **Turbulence** (Standard Deviation spikes) to find bulky isomers like Pseudouridine (pU), and **Current Drops** (Mean shifts) to find charged methylations like m6A.
2. **Multi-Task Learning (MTL):** Once the dataset is mathematically purified, a dual-headed Convolutional Recurrent Neural Network (CRNN) is trained to simultaneously predict the underlying DNA sequence *and* its chemical modification state.

## Repository Structure
```text
NanoSpeech-MTL/
│
├── README.md
├── requirements.txt
├── pipeline.sh                             # Automated end-to-end execution script
│
├── src/                                    # Core Production Pipeline
│   ├── extract_eventalign_features.py      # Parses Nanopolish tabular biophysics
│   ├── denoise_labels.py                   # GMM Physics filter for pU / m6A
│   ├── train_mtl.py                        # Trains the Multi-Task CRNN
│   └── inference_demo.py                   # Live inference script for offline testing
│
└── experimental/                           # Next-Generation Architectures
    ├── r10_dual_head.py                    # 5-kmer extractor for R10 chemistry
    ├── epistasis_gmm.py                    # 4-Class GMM for modification collisions
    └── real_time_unet.py                   # 1D U-Net bypassing EventAlign (2.6M Params)
```

## Installation
We recommend using a Conda environment to manage dependencies:

```bash
conda create -n NanoSpeech python=3.10
conda activate NanoSpeech
pip install -r requirements.txt
```

## Running the Pipeline (NanoBaseLib Integration)
This pipeline is modular and designed to scale across the multi-species datasets provided by **NanoBaseLib**. You can run the entire pipeline automatically using `./pipeline.sh`, or execute the steps manually:

**Step 1: Extract Tabular Biophysical Features** Parses the basecalled FASTQ and Nanopolish `eventalign.txt` to extract the 9-feature sliding window (Mean, Stdv, Dwell).
```bash
python src/extract_eventalign_features.py \
    --fastq data/pass.fastq \
    --eventalign data/eventalign.txt \
    --target_base T \
    --output data/processed/tabular_features.csv
```

**Step 2: Unsupervised Physics Denoising** Cleans the stoichiometric label noise. Use `--mod_type pU` for turbulence-based modifications, or `--mod_type m6A` for current-drop modifications.
```bash
python src/denoise_labels.py \
    --input data/raw/full_production_dataset.pt \
    --mod_type pU \
    --output data/processed/clean_dataset.pt
```

**Step 3: Train the Multi-Task CRNN** Trains the dual-headed network to optimize both the CTC Basecalling Loss and the Binary Cross-Entropy Modification Loss.
```bash
python src/train_mtl.py \
    --dataset data/processed/clean_dataset.pt \
    --epochs 50 \
    --batch_size 1024 \
    --out_weights data/weights/nanospeech_mtl_best.pth
```

**Step 4: Live Inference** Loads the trained weights and processes a raw dataset for offline predictions. Outputs a read-by-read CSV report.
```bash
python src/inference_demo.py \
    --dataset data/processed/clean_dataset.pt \
    --weights data/weights/nanospeech_mtl_best.pth \
    --num_samples 5000 \
    --output results/final_predictions.csv
```

## 📊 Performance Benchmarks
*(Extensive evaluations across NanoBaseLib's multi-species datasets are currently underway. A comprehensive performance table will be added here upon completion.)*

## 🔭 Experimental Avenues (Future Work)
The `experimental/` folder contains prototypes designed to overcome current sequencing limitations:
* **R10 Dual-Head Physics:** Expands the biophysical capture window from 3 to 5 k-mers to handle Oxford Nanopore's newer dual-reader head chemistry.
* **Epistasis Resolution:** A 4-component GMM designed to untangle overlapping physical signatures (e.g., when a bulky pU sits directly adjacent to a charged m6A).
* **Real-Time U-Net (Turbulence Radar):** A 2.6-million parameter 1D U-Net that slides directly over raw continuous `.fast5` electrical waves. This bypasses the computational bottleneck of Nanopolish `eventalign` to allow real-time modification detection during live sequencing.

## 👥 Authors
* **Gülşen Sabak**
* **Süleyman Emir Taşan**
* **Hüseyin Emir Akdağ**
