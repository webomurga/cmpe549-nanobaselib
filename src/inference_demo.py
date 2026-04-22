import argparse
import torch
import torch.nn as nn
import pandas as pd

# --- ARCHITECTURE (Perfectly matched to the trained weights) ---
class NanoSpeechMTL(nn.Module):
    def __init__(self, num_base_classes=4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        self.base_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_base_classes)
        )
        
        self.mod_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Restored for strict PyTorch state_dict matching
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x[:, :, 200:600] 
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_feature = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        base_logits = self.base_head(final_feature)
        mod_logits = self.mod_head(final_feature)
        
        return base_logits, mod_logits

def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Live Inference Demo")
    parser.add_argument("--dataset", required=True, help="Path to the .pt dataset file")
    parser.add_argument("--weights", required=True, help="Path to the trained .pth model weights")
    parser.add_argument("--output", default="nanospeech_predictions.csv", help="Output CSV filename")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of reads to process for the demo")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing NanoSpeech MTL on {device.type.upper()}...")
    model = NanoSpeechMTL().to(device)
    
    print(f"Loading trained weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Loading {args.num_samples} samples from dataset...")
    # Load dataset and take a slice for the live demo
    loaded_data = torch.load(args.dataset, map_location=device, weights_only=True)
    X_all = loaded_data[0]
    X_demo = X_all[:args.num_samples].to(device)
    
    print("Running Deep Learning Inference...")
    with torch.no_grad():
        base_logits, mod_logits = model(X_demo)
        
        # Modification Predictions
        mod_probabilities = torch.sigmoid(mod_logits).squeeze().cpu().numpy()
        mod_predictions = (mod_probabilities > 0.5).astype(int)
        
        # Basecalling Predictions
        base_predictions = torch.argmax(base_logits, dim=1).cpu().numpy()
    
    print(f"Compiling results to {args.output}...")
    
    read_ids = [f"read_{i:05d}" for i in range(args.num_samples)]
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    predicted_bases = [base_map[b] for b in base_predictions]
    
    df = pd.DataFrame({
        'Read_ID': read_ids,
        'Predicted_Base': predicted_bases,
        'Modification_Probability': mod_probabilities,
        'Predicted_Class': mod_predictions
    })
    
    df['Predicted_Class'] = df['Predicted_Class'].map({1: 'Modified', 0: 'Unmodified'})
    df.to_csv(args.output, index=False)
    
    modified_count = sum(mod_predictions)
    print("="*40)
    print("🔬 INFERENCE REPORT")
    print("="*40)
    print(f"Total Reads Processed : {args.num_samples}")
    print(f"Unmodified Found      : {args.num_samples - modified_count}")
    print(f"Modified Found        : {modified_count}")
    print("="*40)
    print(f"Demo complete! Open '{args.output}' to view the read-by-read analysis.")

if __name__ == "__main__":
    main()
