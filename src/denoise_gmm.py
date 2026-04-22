import argparse
import torch
import numpy as np
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Unsupervised Physics Denoising")
    parser.add_argument("--input", default="raw_features.pt", help="Input raw tensor file")
    parser.add_argument("--mod_type", choices=['pU', 'm6A'], required=True, help="Physics toggle")
    parser.add_argument("--out", default="clean_dataset.pt", help="Output clean tensor file")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading raw features from {args.input}...")
    X_all, y_base_all = torch.load(args.input)
    
    print(f"Applying Physics Filter for: {args.mod_type}")
    center_window = X_all[:, 0, 350:450].numpy()
    
    if args.mod_type == 'pU':
        # pU Physics: Bulky isomer causes Turbulence
        physics_feature = np.std(center_window, axis=1).reshape(-1, 1)
        higher_is_modified = True 
    else:
        # m6A Physics: Charged methyl group causes Current Drop
        physics_feature = np.mean(center_window, axis=1).reshape(-1, 1)
        higher_is_modified = False # Lower mean = m6A
        
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(physics_feature)
    
    mean_0, mean_1 = np.mean(physics_feature[cluster_labels == 0]), np.mean(physics_feature[cluster_labels == 1])
    
    if higher_is_modified:
        y_mod_clean = torch.tensor(cluster_labels if mean_1 > mean_0 else 1 - cluster_labels, dtype=torch.float32)
    else:
        y_mod_clean = torch.tensor(cluster_labels if mean_1 < mean_0 else 1 - cluster_labels, dtype=torch.float32)
        
    print(f"Clean Ground Truth Established: {int(len(y_mod_clean) - y_mod_clean.sum())} Normal | {int(y_mod_clean.sum())} Modified")
    
    torch.save((X_all, y_base_all, y_mod_clean), args.out)
    print(f"Clean dataset saved to {args.out}")

if __name__ == "__main__":
    main()
