import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from torchvision.models import vgg19
import numpy as np
import random
import matplotlib.pyplot as plt

#Total Variation Loss
#If neighboring pixels are very different, the image has sharp variations, which could be noise. TV Loss penalizes these variations to smooth the image
#TV Loss is NOT part of adversarial loss, perceptual loss, or MSE loss directly but acts as a regularization loss.
class TV_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tv_height = torch.pow(x[:,:,1:,:] - x[:,:,:-1, :], 2).sum() #e.g: (50 - 48)² + (52 - 49)² + (51 - 50)²
        tv_width = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
        return (tv_height + tv_width)
    
class FeatureExtractor(nn.Module):
    #When you pass an image, it outputs a smaller, processed version that captures key details like edges, textures, and patterns.
    #Instead of comparing images pixel by pixel, we compare their features (important patterns).
    def __init__(self):
        super().__init__()
        vggnet = vgg19(pretrained=True)
        # selects only the first 36 layers.
        #Since we are only extracting features and not training VGG-19, we want to keep the model in evaluation mode.
        #MaxPool layer (index 36) is excluded because it reduces the resolution too much, which is not ideal for Perceptual Loss in SRGAN.
        self.feature_extractor = nn.Sequential(*list(vggnet.features)[:36]).eval() 
        
        # ensures that VGG-19's weights do not change during training.
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)
    

class DIV2KTrainSet(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=128, upscale_factor=4, num_patches=30):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.num_patches = num_patches
        self.hr_files = self._load_file_names(hr_dir)
        self.lr_files = self._load_file_names(lr_dir)

        # Define transformations
        self.hr_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.patch_size // self.upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def _load_file_names(self, directory):
        """Loads file names and ensures they are sorted properly."""
        return sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    def _random_crop(self, image, num_patches):
        """Randomly crops `num_patches` patches from an image."""
        width, height = image.size
        patches = []
        for _ in range(num_patches):
            x = random.randint(0, width - self.patch_size)
            y = random.randint(0, height - self.patch_size)
            patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            patches.append(patch)
        return patches

    def __len__(self):
        return len(self.hr_files) * self.num_patches

    def __getitem__(self, index):
        """Returns one (HR, LR) patch pair."""
        # Determine which image to use
        img_index = index // self.num_patches
        hr_image = Image.open(self.hr_files[img_index]).convert("RGB")

        # Get all 5 HR patches from this image
        hr_patches = self._random_crop(hr_image, self.num_patches)

        # Convert all HR patches to LR patches
        lr_patches = [self.lr_transform(hr_patch) for hr_patch in hr_patches]

        # Pick one patch pair from the 5 (ensures dataset length matches calculation)
        patch_idx = index % self.num_patches
        hr_patch = self.hr_normalize(hr_patches[patch_idx])  # Normalize HR patch
        lr_patch = lr_patches[patch_idx]  # LR is already tensor

        return hr_patch, lr_patch
#hr_patch.shape  # torch.Size([3, 128, 128])  - HR patch
#lr_patch.shape  # torch.Size([3, 32, 32])   - LR patch (downscaled)


class DIV2KValidSet(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=128):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size  # Fixed crop size

        self.hr_files = self._load_file_names(hr_dir)
        self.lr_files = self._load_file_names(lr_dir)

        # Define transformations
        self.hr_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_file_names(self, directory):
        """Load and sort file names to ensure HR and LR images match."""
        return sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    def _get_transforms(self):
        """Define transforms for center cropping HR and LR images."""
        hr_transform = transforms.CenterCrop(self.crop_size)
        lr_transform = transforms.CenterCrop(self.crop_size // 4)  # Assuming x4 downscale
        return hr_transform, lr_transform

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, index):
        """Load matching HR and LR images, apply cropping, and return tensors."""
        hr_image = Image.open(self.hr_files[index]).convert("RGB")
        lr_image = Image.open(self.lr_files[index]).convert("RGB")

        hr_transform, lr_transform = self._get_transforms()

        hr_crop = hr_transform(hr_image)  # Center crop HR
        lr_crop = lr_transform(lr_image)  # Center crop LR

        hr_crop = self.hr_normalize(hr_crop)  # Normalize HR
        lr_crop = transforms.ToTensor()(lr_crop)  # Convert LR to tensor

        return hr_crop, lr_crop
#hr_crop.shape  # torch.Size([3, 128, 128])  - HR crop
#lr_crop.shape  # torch.Size([3, 32, 32])   - LR crop


def save_plot(generator_losses, discriminator_losses, PSNR_valid, mode):
    # Convert lists of tuples to NumPy arrays
    generator_losses = np.array(generator_losses)
    PSNR_valid = np.array(PSNR_valid)

    # Extract epochs and values
    epochs_gen = generator_losses[:, 0]
    loss_gen = generator_losses[:, 1]

    if mode == "adversarial":
        discriminator_losses = np.array(discriminator_losses)
        epochs_disc = discriminator_losses[:, 0]
        loss_disc = discriminator_losses[:, 1]
        loss_filename = "adv_loss.png"
        psnr_filename = "adv_psnr.png"
    else:  # mode == "generator" (pre-training)
        epochs_disc = epochs_gen
        loss_disc = np.zeros_like(loss_gen)  # No discriminator loss in generator-only training
        loss_filename = "pretrain_loss.png"
        psnr_filename = "pretrain_psnr.png"

    epochs_psnr = PSNR_valid[:, 0]
    psnr_values = PSNR_valid[:, 1]

    # Plot 1: Generator & Discriminator Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_gen, loss_gen, label="Generator Loss", marker='o', linestyle='-')
    if mode == "adversarial":
        plt.plot(epochs_disc, loss_disc, label="Discriminator Loss", marker='s', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{mode.capitalize()} Training - Generator & Discriminator Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'C:/Users/User/Desktop/SRGAN/div2k_train_val/outputs/{loss_filename}')
    plt.close()

    # Plot 2: PSNR
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_psnr, psnr_values, label="PSNR", marker='^', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title(f"{mode.capitalize()} Training - PSNR Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'C:/Users/User/Desktop/SRGAN/div2k_train_val/outputs/{psnr_filename}')
    plt.close()


