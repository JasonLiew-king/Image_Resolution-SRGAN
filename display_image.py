import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from skimage.metrics import structural_similarity as compare_ssim
from model import Generator  # Assuming SRGAN's generator is implemented

def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--original", type=str, required=True)
parser.add_argument("--bicubic", type=str, required=True)
parser.add_argument("--srgan", type=str, required=True)
parser.add_argument("--patch_info", nargs='+', type=int, required=True,
                    help="List of patches: x1 y1 size1 x2 y2 size2 ...")
args = parser.parse_args()

# Load and normalize images
original = cv2.cvtColor(cv2.imread(args.original), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
bicubic = cv2.cvtColor(cv2.imread(args.bicubic), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
srgan = cv2.cvtColor(cv2.imread(args.srgan), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Convert to tensors
original_torch = torch.tensor(original).permute(2, 0, 1)
bicubic_torch = torch.tensor(bicubic).permute(2, 0, 1)
srgan_torch = torch.tensor(srgan).permute(2, 0, 1)

# Compute metrics
psnr_bicubic = calc_psnr(original_torch, bicubic_torch).item()
mse_bicubic = calc_mse(original_torch, bicubic_torch).item()
psnr_srgan = calc_psnr(original_torch, srgan_torch).item()
mse_srgan = calc_mse(original_torch, srgan_torch).item()

# SSIM
original_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
bicubic_gray = cv2.cvtColor((bicubic * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
srgan_gray = cv2.cvtColor((srgan * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
ssim_bicubic = compare_ssim(original_gray, bicubic_gray, data_range=255)
ssim_srgan = compare_ssim(original_gray, srgan_gray, data_range=255)

# Parse patch info: every 3 values = (x, y, size)
info = args.patch_info
patches_original, patches_bicubic, patches_srgan = [], [], []
coords_sizes = [(info[i], info[i+1], info[i+2]) for i in range(0, len(info), 3)]

colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0.5, 1, 0)]  # red, orange, yellow, lime

for idx, (x, y, size) in enumerate(coords_sizes):
    if x + size > original.shape[1] or y + size > original.shape[0]:
        print(f"Skipping patch at ({x},{y}) with size {size}: out of bounds.")
        continue

    patch_o = original[y:y+size, x:x+size]
    patch_b = bicubic[y:y+size, x:x+size]
    patch_s = srgan[y:y+size, x:x+size]

    patches_original.append(patch_o)
    patches_bicubic.append(patch_b)
    patches_srgan.append(patch_s)

    # Draw rectangles
    color = colors[idx % len(colors)]
    for img in [original, bicubic, srgan]:
        cv2.rectangle(img, (x, y), (x + size, y + size), color, 7)

# Plotting
num_patches = len(patches_original)
fig, axs = plt.subplots(num_patches + 1, 3, figsize=(15, 4 * (num_patches + 1)))

# Display full images with patch boxes
axs[0, 0].imshow(original)
axs[0, 0].set_title("Original", fontsize=14, color="orange")
axs[0, 0].axis("off")

axs[0, 1].imshow(bicubic)
axs[0, 1].set_title(f"Bicubic\nPSNR: {psnr_bicubic:.2f} dB | SSIM: {ssim_bicubic:.4f} | MSE: {mse_bicubic:.4f}", fontsize=14, color="orange")
axs[0, 1].axis("off")

axs[0, 2].imshow(srgan)
axs[0, 2].set_title(f"SRGAN\nPSNR: {psnr_srgan:.2f} dB | SSIM: {ssim_srgan:.4f} | MSE: {mse_srgan:.4f}", fontsize=14, color="orange")
axs[0, 2].axis("off")

# Plot patches row by row
for i in range(num_patches):
    axs[i+1, 0].imshow(patches_original[i])
    axs[i+1, 0].set_title(f"Patch {i+1} (Original)", fontsize=12, color="orange")
    axs[i+1, 0].axis("off")

    axs[i+1, 1].imshow(patches_bicubic[i])
    axs[i+1, 1].set_title(f"Patch {i+1} (Bicubic)", fontsize=12, color="orange")
    axs[i+1, 1].axis("off")

    axs[i+1, 2].imshow(patches_srgan[i])
    axs[i+1, 2].set_title(f"Patch {i+1} (SRGAN)",fontsize=12, color="orange")
    axs[i+1, 2].axis("off")

plt.tight_layout()
plt.show()
