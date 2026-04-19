
import torch
import torch.nn as nn
import torch.nn.functional as F

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss 1: Spatial Consistency Loss
# Makes sure the enhanced image looks similar to input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        left_kernel  = torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        right_kernel = torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        up_kernel    = torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        down_kernel  = torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('left_kernel',  left_kernel)
        self.register_buffer('right_kernel', right_kernel)
        self.register_buffer('up_kernel',    up_kernel)
        self.register_buffer('down_kernel',  down_kernel)

    def forward(self, org, enhance):
        org_mean     = torch.mean(org,     1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool     = F.avg_pool2d(org_mean,     4)
        enhance_pool = F.avg_pool2d(enhance_mean, 4)

        d_org_left   = F.conv2d(org_pool,     self.left_kernel,  padding=1)
        d_org_right  = F.conv2d(org_pool,     self.right_kernel, padding=1)
        d_org_up     = F.conv2d(org_pool,     self.up_kernel,    padding=1)
        d_org_down   = F.conv2d(org_pool,     self.down_kernel,  padding=1)

        d_enh_left   = F.conv2d(enhance_pool, self.left_kernel,  padding=1)
        d_enh_right  = F.conv2d(enhance_pool, self.right_kernel, padding=1)
        d_enh_up     = F.conv2d(enhance_pool, self.up_kernel,    padding=1)
        d_enh_down   = F.conv2d(enhance_pool, self.down_kernel,  padding=1)

        loss = (torch.pow(d_org_left  - d_enh_left,  2) +
                torch.pow(d_org_right - d_enh_right, 2) +
                torch.pow(d_org_up    - d_enh_up,    2) +
                torch.pow(d_org_down  - d_enh_down,  2))
        return torch.mean(loss)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss 2: Exposure Control Loss
# Makes sure image is not too dark or too bright
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x_mean = torch.mean(x, 1, keepdim=True)
        mean   = self.pool(x_mean)
        return torch.mean(torch.pow(mean - self.mean_val, 2))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss 3: Color Constancy Loss
# Keeps colors balanced, avoids color cast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ColorConstancyLoss(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:,0], mean_rgb[:,1], mean_rgb[:,2]
        return torch.mean(torch.pow(mr - mg, 2) +
                          torch.pow(mr - mb, 2) +
                          torch.pow(mg - mb, 2))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss 4: Illumination Smoothness Loss
# Makes enhancement smooth, avoids sharp artifacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, x):
        tv_h = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
        return (tv_h + tv_w) / (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == '__main__':
    dummy_org = torch.randn(2, 3, 256, 256)
    dummy_enh = torch.randn(2, 3, 256, 256)

    L_spa = SpatialConsistencyLoss()
    L_exp = ExposureControlLoss()
    L_col = ColorConstancyLoss()
    L_ill = IlluminationSmoothnessLoss()

    print(f"✅ Spatial Consistency Loss : {L_spa(dummy_org, dummy_enh).item():.4f}")
    print(f"✅ Exposure Control Loss    : {L_exp(dummy_enh).item():.4f}")
    print(f"✅ Color Constancy Loss     : {L_col(dummy_enh).item():.4f}")
    print(f"✅ Illumination Smoothness  : {L_ill(dummy_enh).item():.4f}")
    print("🎉 All loss functions working!")
