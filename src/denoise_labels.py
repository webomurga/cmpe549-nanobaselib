import argparse
import torch
import numpy as np
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Unsupervised Physics Denoising (GMM)")
    parser.add_argument("--input", required=True, help="Path to raw tensor dataset (.pt file)")
    parser.add_argument("--mod_type", choices=['pU', 'm6A'], required=True, help="Physics toggle: pU (Turbulence) or m6A (Current Drop)")
    parser.add_argument("--output", default="clean_dataset.pt", help="Output path for the denoised tensor dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading raw continuous tensor dataset from {args.input}...")
    # Assumes input tensor format: (X_all, y_base_all, [optional noisy labels])
    loaded_data = torch.load(args.input, weights_only=True)
    X_all = loaded_data[0]
    y_base_all = loaded_data[1]
    
    print(f"\nDiscovering clean labels via Physics Mode: {args.mod_type}...")
    
    # Slice the exact center of the squiggle where the nanopore is reading the target base
    center_window = X_all[:, 0, 350:450].numpy()
    
    if args.mod_type == 'pU':
        # pU Physics: Bulky isomer causes high Standard Deviation (Turbulence)
        print("   -> Extracting Signal Variance (Turbulence)...")
        physics_feature = np.std(center_window, axis=1).reshape(-1, 1)
        higher_is_modified = True
    elif args.mod_type == 'm6A':
        # m6A Physics: Charged methyl group causes massive drop in Mean Current
        print("   -> Extracting Signal Mean (Current Drop)...")
        physics_feature = np.mean(center_window, axis=1).reshape(-1, 1)
        higher_is_modified = False 

    # Run the GMM to separate the physical populations
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(physics_feature)
    
    mean_0 = np.mean(physics_feature[cluster_labels == 0])
    mean_1 = np.mean(physics_feature[cluster_labels == 1])
    
    if higher_is_modified:
        if mean_1 > mean_0:
            y_mod_clean = torch.tensor(cluster_labels, dtype=torch.float32)
        else:
            y_mod_clean = torch.tensor(1 - cluster_labels, dtype=torch.float32)
    else:
        if mean_1 < mean_0:
            y_mod_clean = torch.tensor(cluster_labels, dtype=torch.float32)
        else:
            y_mod_clean = torch.tensor(1 - cluster_labels, dtype=torch.float32)

    num_modified = int(y_mod_clean.sum().item())
    num_unmodified = len(y_mod_clean) - num_modified
    
    print(f"   Clean Ground Truth Established:")
    print(f"      Unmodified Reads: {num_unmodified}")
    print(f"      Modified Reads  : {num_modified}")

    print(f"\nSaving purified dataset to {args.output}...")
    torch.save((X_all, y_base_all, y_mod_clean), args.output)
    print("Denoising Complete!")

if __name__ == "__main__":
    main()
