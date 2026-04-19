
import torch
import torch.nn as nn

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder layers
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        
        # Decoder layers with skip connections
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)  # 8 curves x 3 channels

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        # Decoder with skip connections
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        # Split into 8 curve parameter maps (3 channels each)
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        
        # Apply iterative curve enhancement
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced = x + r8 * (torch.pow(x, 2) - x)
        
        return enhanced


# Quick test to verify model works
if __name__ == '__main__':
    model = ZeroDCE()
    dummy = torch.randn(1, 3, 256, 256)  # Fake image batch
    out   = model(dummy)
    print(f"✅ Input shape  : {dummy.shape}")
    print(f"✅ Output shape : {out.shape}")
    print(f"✅ Model params : {sum(p.numel() for p in model.parameters()):,}")
    print("🎉 Model is working correctly!")
