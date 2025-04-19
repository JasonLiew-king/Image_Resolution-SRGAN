import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from skimage.metrics import structural_similarity as compare_ssim


# -------------------------------
# Metric Functions
# -------------------------------
def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 100 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).mean()

def center_crop(img, target_h, target_w):
    h, w = img.shape[:2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[top:top + target_h, left:left + target_w]


# -------------------------------
# Argument Parser
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--original", type=str, required=True)
parser.add_argument("--bicubic", type=str, required=True)
parser.add_argument("--srresnet", type=str, required=True)
parser.add_argument("--srgan", type=str, required=True)
parser.add_argument("--patch_info", nargs='+', type=int, required=True,
                    help="List of patches: x1 y1 size1 x2 y2 size2 ...")
args = parser.parse_args()


# -------------------------------
# Load & Preprocess Images
# -------------------------------
def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img

original = load_image(args.original)
bicubic = load_image(args.bicubic)
srresnet = load_image(args.srresnet)
srgan = load_image(args.srgan)

# Ensure all images are same size via center crop
min_h = min(original.shape[0], bicubic.shape[0], srresnet.shape[0], srgan.shape[0])
min_w = min(original.shape[1], bicubic.shape[1], srresnet.shape[1], srgan.shape[1])

original = center_crop(original, min_h, min_w)
bicubic = center_crop(bicubic, min_h, min_w)
srresnet = center_crop(srresnet, min_h, min_w)
srgan = center_crop(srgan, min_h, min_w)

# Convert to tensors
original_torch = torch.tensor(original).permute(2, 0, 1)
bicubic_torch = torch.tensor(bicubic).permute(2, 0, 1)
srresnet_torch = torch.tensor(srresnet).permute(2, 0, 1)
srgan_torch = torch.tensor(srgan).permute(2, 0, 1)


# -------------------------------
# Compute Metrics
# -------------------------------
def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return compare_ssim(img1_gray, img2_gray, data_range=255)

# PSNR / SSIM / MSE
psnr_bicubic = calc_psnr(original_torch, bicubic_torch).item()
psnr_srresnet = calc_psnr(original_torch, srresnet_torch).item()
psnr_srgan = calc_psnr(original_torch, srgan_torch).item()

mse_bicubic = calc_mse(original_torch, bicubic_torch).item()
mse_srresnet = calc_mse(original_torch, srresnet_torch).item()
mse_srgan = calc_mse(original_torch, srgan_torch).item()

ssim_bicubic = compute_ssim(original, bicubic)
ssim_srresnet = compute_ssim(original, srresnet)
ssim_srgan = compute_ssim(original, srgan)


# -------------------------------
# Patch Extraction & Drawing
# -------------------------------
info = args.patch_info
patch_coords = [(info[i], info[i+1], info[i+2]) for i in range(0, len(info), 3)]
colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0.5, 1, 0)]  # red to lime

patches_original, patches_bicubic, patches_srresnet, patches_srgan = [], [], [], []

for idx, (x, y, size) in enumerate(patch_coords):
    if x + size > original.shape[1] or y + size > original.shape[0]:
        print(f"Skipping patch at ({x}, {y}) with size {size}: out of bounds.")
        continue

    patches_original.append(original[y:y+size, x:x+size])
    patches_bicubic.append(bicubic[y:y+size, x:x+size])
    patches_srresnet.append(srresnet[y:y+size, x:x+size])
    patches_srgan.append(srgan[y:y+size, x:x+size])

    # Draw rectangles for visualization
    color = colors[idx % len(colors)]
    for img in [original, bicubic, srresnet, srgan]:
        cv2.rectangle(img, (x, y), (x + size, y + size), color, 7)


# -------------------------------
# Plotting
# -------------------------------
num_patches = len(patches_original)
fig, axs = plt.subplots(num_patches + 1, 4, figsize=(20, 4 * (num_patches + 1)))

titles = [
    "Original",
    f"Bicubic\nPSNR: {psnr_bicubic:.2f} dB | SSIM: {ssim_bicubic:.4f} | MSE: {mse_bicubic:.4f}",
    f"SRResNet\nPSNR: {psnr_srresnet:.2f} dB | SSIM: {ssim_srresnet:.4f} | MSE: {mse_srresnet:.4f}",
    f"SRGAN\nPSNR: {psnr_srgan:.2f} dB | SSIM: {ssim_srgan:.4f} | MSE: {mse_srgan:.4f}"
]
images = [original, bicubic, srresnet, srgan]

for i in range(4):
    axs[0, i].imshow(images[i])
    axs[0, i].set_title(titles[i], fontsize=14, color="orange")
    axs[0, i].axis("off")

# Patches
for i in range(num_patches):
    axs[i+1, 0].imshow(patches_original[i])
    axs[i+1, 0].set_title(f"Patch {i+1} (Original)", fontsize=12, color="orange")
    axs[i+1, 0].axis("off")

    axs[i+1, 1].imshow(patches_bicubic[i])
    axs[i+1, 1].set_title(f"Patch {i+1} (Bicubic)", fontsize=12, color="orange")
    axs[i+1, 1].axis("off")

    axs[i+1, 2].imshow(patches_srresnet[i])
    axs[i+1, 2].set_title(f"Patch {i+1} (SRResNet)", fontsize=12, color="orange")
    axs[i+1, 2].axis("off")

    axs[i+1, 3].imshow(patches_srgan[i])
    axs[i+1, 3].set_title(f"Patch {i+1} (SRGAN)", fontsize=12, color="orange")
    axs[i+1, 3].axis("off")

plt.tight_layout()
plt.show()
