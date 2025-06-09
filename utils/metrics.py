import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex').eval()
if torch.cuda.is_available():
    loss_fn_alex = loss_fn_alex.cuda()

def compute_rgb_relative_error(original, compressed):
    return np.mean([
        np.linalg.norm(original[:, :, i] - compressed[:, :, i], ord='fro') /
        np.linalg.norm(original[:, :, i], ord='fro')
        for i in range(3)
    ])

def compute_rgb_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def compute_rgb_ssim(original, compressed):
    o = original.astype(np.float64) / 255.0
    c = compressed.astype(np.float64) / 255.0
    return ssim(o, c, channel_axis=2, data_range=1.0)

def compute_rgb_lpips(original, compressed):
    def to_tensor(image):
        image = torch.from_numpy(image).float() / 255.0 * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image.cuda() if torch.cuda.is_available() else image
    with torch.no_grad():
        return loss_fn_alex(to_tensor(original), to_tensor(compressed)).item()
