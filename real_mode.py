import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from real_dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import psutil


def log_memory(stage, device):
    ram = psutil.virtual_memory()
    print(f"[{stage}] RAM Used: {(ram.total - ram.available) / 1024 ** 2:.2f} MB / {ram.total / 1024 ** 2:.2f} MB")

    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB")


def train_generator_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    log_memory("Start", device)

    # ---- TRAINING SETUP ----
    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])  #RAM
    train_dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform) #RAM
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) #RAM

    # ---- VALIDATION SETUP ----
    #RAM
    val_dataset = mydata(GT_path=args.val_GT_path, LR_path=args.val_LR_path, in_memory=False, transform=None) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)
    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        print("Pre-trained model loaded from:", args.generator_path)

    generator = generator.to(device)  #GPU
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    for pre_epoch in range(args.pre_train_epoch):
        generator.train()
        log_memory(f"Epoch {pre_epoch+1} - Training Start", device)

        for tr_data in tqdm(train_loader, desc=f'Training Epoch {pre_epoch+1}/{args.pre_train_epoch}'):
            gt = tr_data['GT'].to(device) #GPU
            lr = tr_data['LR'].to(device) #GPU

            output, _ = generator(lr) #GPU
            loss = l2_loss(gt, output) #GPU

            g_optim.zero_grad()
            loss.backward()    # GPU
            g_optim.step()

        print(f"[Epoch {pre_epoch+1}] Training Loss: {loss.item():.6f}")
        log_memory(f"Epoch {pre_epoch+1} - Training End", device)

        # ---- VALIDATION ----
        generator.eval()
        psnr_list = []
        with torch.no_grad():
            for te_data in tqdm(val_loader, desc=f'Validation Epoch {pre_epoch+1}/{args.pre_train_epoch}', unit='img'):
                gt = te_data['GT'].to(device)
                lr = te_data['LR'].to(device)

                bs, c, h, w = lr.size()
                gt = gt[:, :, : h * args.scale, : w * args.scale]

                output, _ = generator(lr)

                output = output[0].cpu().numpy()
                output = np.clip(output, -1.0, 1.0)
                gt = gt[0].cpu().numpy()

                output = (output + 1.0) / 2.0
                gt = (gt + 1.0) / 2.0

                output = output.transpose(1, 2, 0)
                gt = gt.transpose(1, 2, 0)

                y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
                y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

                psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
                psnr_list.append(psnr)

        avg_psnr = np.mean(psnr_list)
        print(f"[Epoch {pre_epoch+1}] Validation PSNR: {avg_psnr:.4f}")
        log_memory(f"Epoch {pre_epoch+1} - Validation End", device)

        # Save model
        save_path = f'C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/generator/weights/pre_trained_model_{pre_epoch+1:03d}.pt'
        torch.save(generator.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        
        

def train_adversarial(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #print(f"Using device: {device}")
    #log_memory("Start", device)
    
    #Training setup
    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # ---- VALIDATION SETUP ----
    #RAM
    val_dataset = mydata(GT_path=args.val_GT_path, LR_path=args.val_LR_path, in_memory=False, transform=None) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)
    #if args.fine_tuning:
    #    generator.load_state_dict(torch.load(args.generator_path))
    #    print("pre-trained model is loaded")
    #    print("path : %s" % (args.generator_path))
    generator.load_state_dict(torch.load(args.generator_path))  # must load pretrained
    print("pre-trained model is loaded")
    generator = generator.to(device)
    #generator.train()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale).to(device)
    #discriminator.train()
    
    #log_memory("After Model Initialization", device) #check memory

    vgg_net = vgg19().to(device).eval()
    l2_loss = nn.MSELoss()
    VGG_loss = perceptual_loss(vgg_net)
    tv_loss = TVLoss()
    cross_ent = nn.BCELoss()

    g_optim = optim.Adam(generator.parameters(), lr=1e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    fine_epoch = 0
    while fine_epoch < args.fine_train_epoch:
        log_memory(f"Start Training Epoch {fine_epoch+1}", device) #check memory
        scheduler.step()
        
        # Training Loop
        generator.train()  # Ensure we are in training mode for generator
        discriminator.train()  # Ensure we are in training mode for discriminator

        for tr_data in tqdm(loader, desc=f'Adversarial Epoch {fine_epoch+1}/{args.fine_train_epoch}'):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            ## Train Discriminator
            #You're assuming generator(lr) returns a tensor (e.g., an image), but it actually returns two things — maybe (output, features) or something similar.
            #output = generator(lr).detach()
            # Train Discriminator
            output, _ = generator(lr)
            output = output.detach()
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            ## Train Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)

            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer)

            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            #log_memory("After Training Batch", device) #check memory

        fine_epoch += 1

        print(f"Adversarial Epoch {fine_epoch}, G Loss: {g_loss.item():.6f}, D Loss: {d_loss.item():.6f}")
        #log_memory(f"After Training Epoch {fine_epoch+1}", device)

         # ---- VALIDATION ----
        generator.eval()  # Switch to eval mode for validation
        psnr_list = []
        with torch.no_grad():
            for te_data in tqdm(val_loader, desc=f'Validation Epoch {fine_epoch}/{args.fine_train_epoch}', unit='img'):
                gt = te_data['GT'].to(device)
                lr = te_data['LR'].to(device)

                bs, c, h, w = lr.size()
                gt = gt[:, :, : h * args.scale, : w * args.scale]

                output, _ = generator(lr)

                output = output[0].cpu().numpy()
                output = np.clip(output, -1.0, 1.0)
                gt = gt[0].cpu().numpy()

                output = (output + 1.0) / 2.0
                gt = (gt + 1.0) / 2.0

                output = output.transpose(1, 2, 0)
                gt = gt.transpose(1, 2, 0)

                y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
                y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

                psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
                psnr_list.append(psnr)

        avg_psnr = np.mean(psnr_list)
        print(f"[Epoch {fine_epoch}] Validation PSNR: {avg_psnr:.4f}")
        #log_memory(f"After Validation Epoch {fine_epoch}", device)  #check memory
        
        torch.save(generator.state_dict(), f'C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/adverserial/outputs/SRGAN_gene_{fine_epoch:03d}.pt')
        torch.save(discriminator.state_dict(), f'C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/adverserial/outputs/SRGAN_discrim_{fine_epoch:03d}.pt')
       # log_memory(f"After Saving Models Epoch {fine_epoch}", device) #check memory







# In[ ]:

def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    f = open('C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/test/result.txt', 'w')
    psnr_list = []
    
    with torch.no_grad():
        for i, te_data in enumerate(tqdm(loader, desc="Testing", unit="img")):  # wrapped with tqdm here
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            
            psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/test/img/res_%04d.png' % i)

        f.write('avg psnr : %04f' % np.mean(psnr_list))
