import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Avenue B: Epistasis Collision GMM")
    parser.add_argument("--input", required=True, help="Path to raw continuous tensor dataset (.pt)")
    parser.add_argument("--output", default="epistasis_report.csv", help="Output classification report")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading raw continuous tensor dataset from {args.input}...")
    loaded_data = torch.load(args.input, weights_only=True)
    X_all = loaded_data[0]
    
    print("Running 4-Component Unsupervised Epistasis Discovery...")
    # Extract the center window
    center_window = X_all[:, 0, 350:450].numpy()
    
    # Calculate BOTH Mean and Standard Deviation to capture all physics
    target_mean = np.mean(center_window, axis=1).reshape(-1, 1)
    target_stdv = np.std(center_window, axis=1).reshape(-1, 1)
    
    # Combine into a 2D feature space for the GMM
    target_physics = np.hstack((target_mean, target_stdv))
    
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(target_physics)
    
    # Map mathematically discovered clusters to biophysical realities
    centroids = gmm.means_
    labels_map = {}
    
    global_mean_current = np.mean(centroids[:, 0])
    global_mean_stdv = np.mean(centroids[:, 1])
    
    for i, centroid in enumerate(centroids):
        c_mean, c_stdv = centroid[0], centroid[1]
        
        if c_mean >= global_mean_current and c_stdv < global_mean_stdv:
            labels_map[i] = "Unmodified (U)"
        elif c_mean >= global_mean_current and c_stdv >= global_mean_stdv:
            labels_map[i] = "Pseudouridine (pU) - High Turbulence"
        elif c_mean < global_mean_current and c_stdv < global_mean_stdv:
            labels_map[i] = "Methylation (m6A) - Current Drop"
        else:
            labels_map[i] = "Collision (pU + m6A) - Drop & Turbulence!"
            
    # Apply mapping
    final_labels = [labels_map[label] for label in cluster_labels]
    read_ids = [f"read_{i:06d}" for i in range(len(final_labels))]
    
    df = pd.DataFrame({
        'Read_ID': read_ids,
        'Target_Mean': target_physics[:, 0],
        'Target_Stdv': target_physics[:, 1],
        'Discovered_State': final_labels
    })
    
    print("\nEpistasis Discovery Report:")
    print(df['Discovered_State'].value_counts())
    
    df.to_csv(args.output, index=False)
    print(f"\nEpistasis mapping saved to {args.output}")

if __name__ == "__main__":
    main()
