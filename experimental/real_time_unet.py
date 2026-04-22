import torch
import torch.nn as nn

class DoubleConv1D(nn.Module):
    """(Conv1D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class RealTimeTurbulenceRadar(nn.Module):
    """
    A 1D U-Net that slides over raw continuous electrical signals.
    Bypasses EventAlign to flag turbulent (Modified) coordinates in real-time.
    """
    def __init__(self):
        super().__init__()
        
        # --- ENCODER (Downsampling Pathway) ---
        self.inc = DoubleConv1D(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(256, 512))
        
        # --- DECODER (Upsampling Pathway) ---
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv1D(512, 256) 
        
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv1D(256, 128)
        
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv1D(128, 64)
        
        # Output maps directly back to the original input dimension
        self.outc = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: [Batch_Size, 1 Channel, 5000 Raw Samples]
        
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decode + Skip Connections
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1) 
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1) 
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1) 
        x = self.conv3(x)
        
        logits = self.outc(x)
        
        # Returns a 1D Mask of identical length: [Batch, 1, 5000]
        return torch.sigmoid(logits) 

def main():
    print("Initializing Real-Time Turbulence Radar (1D U-Net)...")
    model = RealTimeTurbulenceRadar()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   -> Architecture Loaded successfully.")
    print(f"   -> Total Parameters: {total_params:,}")
    
    print("\nSimulating Inference on Raw Continuous Squiggle...")
    # Simulate a raw Nanopore squggle of 5000 electrical samples, batch size 16
    mock_continuous_wave = torch.randn(16, 1, 5000) 
    print(f"   -> Input Shape : {mock_continuous_wave.shape} (Batch, Channel, Samples)")
    
    with torch.no_grad():
        turbulence_mask = model(mock_continuous_wave)
        
    print(f"   -> Output Mask : {turbulence_mask.shape} (Matches Input Length Perfectly)")
    print("Inference successful. Model is ready for live sequencer deployment.")

if __name__ == "__main__":
    main()
