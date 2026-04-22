# NanoSpeech-MTL: Biophysical RNA Modification Detector

This repository contains the complete Multi-Task Learning (MTL) pipeline for detecting RNA modifications (like Pseudouridine and m6A) directly from raw Nanopore electrical signals. 

Unlike standard basecallers, this pipeline mathematically purifies the dataset using unsupervised biophysics (Gaussian Mixture Models) to overcome stoichiometric label noise before training a dual-headed CRNN to simultaneously basecall and detect modifications.

## Repository Structure
- `src/1_extract_eventalign_features.py`: Parses Nanopolish eventalign and basecalled FASTQ files to extract the 9-feature tabular biophysics (Mean, Stdv, Dwell).
- `src/2_denoise_labels.py`: Uses Unsupervised GMMs on raw continuous signals to discover true modification states based on molecular physics (Turbulence vs. Current Drop).
- `src/3_train_mtl.py`: Trains the Multi-Task CRNN on the denoised dataset.
- `src/4_inference_demo.py`: Runs live predictions using the trained model weights.

## Requirements
```bash
pip install torch numpy pandas scikit-learn
```
