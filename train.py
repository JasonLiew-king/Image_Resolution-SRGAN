import os
import re
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils import DIV2KTrainSet, DIV2KValidSet, FeatureExtractor, TV_Loss, save_plot
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument('--trainset_dir', type=str, default='C:/Users/User/Desktop/SRGAN/div2k_train_val/train/train_hr/', help='training dataset path')

#parser.add_argument('--validset_dir', type=str, default='C:/Users/User/Desktop/SRGAN/div2k_train_val/val/val_hr/', help='validation dataset path')

parser.add_argument('--train_hr_dir', type=str, 
                    default='C:/Users/User/Desktop/SRGAN/div2k_train_val/train/train_hr/', 
                    help='Path to high-resolution training images')

parser.add_argument('--train_lr_dir', type=str, 
                    default='C:/Users/User/Desktop/SRGAN/div2k_train_val/train/train_lr/', 
                    help='Path to low-resolution training images')

parser.add_argument('--valid_hr_dir', type=str, 
                    default='C:/Users/User/Desktop/SRGAN/div2k_train_val/val/val_hr/', 
                    help='Path to high-resolution validation images')

parser.add_argument('--valid_lr_dir', type=str, 
                    default='C:/Users/User/Desktop/SRGAN/div2k_train_val/val/val_lr/', 
                    help='Path to low-resolution validation images')

parser.add_argument('--upscale_factor', type=int, default=4, choices=[2,3,4,8], help='super resolution upscale factor')

parser.add_argument('--epochs', type=int, default=100, help='training epoch number')

parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training")

parser.add_argument('--mode', type=str, default='adversarial', choices=['adversarial', 'generator'], help='apply adversarial training')

parser.add_argument('--pretrain', type=str, default=None, help='load pretrained generator model')

parser.add_argument('--cuda', action='store_true', help='Using GPU to train')

parser.add_argument('--out_dir', type=str, default='C:/Users/User/Desktop/SRGAN/div2k_train_val/outputs/', help='The path for checkpoints and outputs')

sr_transform = transforms.Compose([
    transforms.Normalize((-1,-1,-1),(2,2,2)),  # Normalization
    #This rescales pixel values from [0,1] (after ToTensor()) back to [0,255] (original image range).
    transforms.ToPILImage()  # Convert Tensor to PIL Image
])

lr_transform = transforms.Compose([
    transforms.ToPILImage()  #Converts a PyTorch Tensor back into a PIL Image.
])



#This ensures the script only runs when executed directly, not when imported as a module.
if __name__ == '__main__':
    opt = parser.parse_args()
    upscale_factor = opt.upscale_factor
    generator_lr = 0.0001
    discriminator_lr = 0.0001
    
    #Creates directories for saving model checkpoints, weights, and output images.
    check_points_dir = opt.out_dir + 'check_points/'
    weights_dir = opt.out_dir + 'weights/'
    imgout_dir = opt.out_dir + 'output/'
    os.makedirs(check_points_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(imgout_dir, exist_ok=True)
    
    # Initialize training dataset with sliding window patches
    train_set = DIV2KTrainSet(
        hr_dir=opt.train_hr_dir,  # High-resolution training directory
        lr_dir=opt.train_lr_dir,  # Low-resolution training directory
        patch_size=128,  # Patch size for training
        stride=98        # Stride for sliding window
    )
    
    # Initialize validation dataset with center-cropped patches
    valid_set = DIV2KValidSet(
        hr_dir=opt.valid_hr_dir,  # High-resolution validation directory
        lr_dir=opt.valid_lr_dir,  # Low-resolution validation directory
        crop_size=128  # Fixed center crop size for validation
    )
    
    # Load training and validation datasets using DataLoader
    #For training the generator
    #trainloader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    #validloader = DataLoader(dataset=valid_set, batch_size=2, shuffle=False)
    
    #For training adverserial
    trainloader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    validloader = DataLoader(dataset=valid_set, batch_size=2, shuffle=False)
        
    #Generator: Creates high-resolution images from low-resolution inputs.
    generator_net = Generator(upscale_factor = upscale_factor, num_blocks=16)
    #Discriminator: Tries to distinguish real high-resolution images from fake ones.
    discriminator_net = Discriminator()
    
    adversarial_criterion = nn.BCELoss()  # Loss for discriminator (Binary Cross Entropy) discriminator is a binary classifier
    content_criterion = nn.MSELoss()      # Loss for generator (Mean Squared Error) train the generator // directly minimizes pixel-wise errors.
    tv_reg = TV_Loss()                    #Total Variation Loss (TV Loss) smooths the image output to reduce artifacts.
    
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=generator_lr)
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=discriminator_lr)
    feature_extractor = FeatureExtractor()
    
    #Moves models and loss functions to GPU for faster training.
    if torch.cuda.is_available() and opt.cuda:
        generator_net.cuda()
        discriminator_net.cuda()
        adversarial_criterion.cuda()
        content_criterion.cuda()
        feature_extractor.cuda()
        
    generator_running_loss = 0.0
    generator_losses = []
    discriminator_losses = []
    PSNR_valid = []
    
    
#==========================================Resume Running Functions===========================================#
    #If training was interrupted, resume from a saved checkpoint.
    #Training was interrupted, and you want to continue from where you left off
    if opt.resume:  
        # If opt.resume is a full path (custom checkpoint file)
        if os.path.isfile(opt.resume):  
            check_point_path = opt.resume  
        else:  
            # Otherwise, assume it's in the default folder
            check_point_path = os.path.join(check_points_dir, f"check_point_epoch_{opt.resume}.pth")  
    
        # Load checkpoint
        check_point = torch.load(check_point_path)  
        generator_net.load_state_dict(check_point['generator'])  
        generator_optimizer.load_state_dict(check_point['generator_optimizer'])  
        generator_losses = check_point['generator_losses']  
        PSNR_valid = check_point['PSNR_valid']  
    
        # Restore Discriminator (if using Adversarial Training)
        if opt.mode == 'adversarial':  
            discriminator_net.load_state_dict(check_point['discriminator'])  
            discriminator_optimizer.load_state_dict(check_point['discriminator_optimizer'])  
            discriminator_losses = check_point['discriminator_losses']  
    
        # Extract epoch number from filename
        #(\d+) means "one or more digits" (like 1, 10, 123).
        #If the pattern is found, match.group(1) retrieves the first part of the match, which is the digits after check_point_epoch_ (e.g., 5 from check_point_epoch_5.pth).
        match = re.search(r'check_point_epoch_(\d+).pth', check_point_path)
        start_epoch = int(match.group(1)) + 1 if match else 1  #add 1 because it is starting from a new epoch
    else:
        start_epoch = 1  # Start from scratch if no checkpoint is provided

    
    #You want to start training from a pre-trained model instead of scratch 
    if opt.pretrain != None:
        saved_G_state = torch.load(opt.pretrain, map_location=torch.device('cuda' if opt.cuda else 'cpu'))
        # generator_net.load_state_dict(saved_G_state['generator'])
        generator_net.load_state_dict(saved_G_state)
        print("Pre-trained weights loaded successfully!")
        
        
#==========================================Pre-train the generator===========================================#
    if opt.mode == 'generator':
        #The training runs from 1 + opt.resume to opt.epochs + 1.
        for epoch in range(start_epoch, opt.epochs+1):  
            print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
            generator_net.train() #Puts the generator in training mode (important for layers like BatchNorm and Dropout).
            training_bar = tqdm(trainloader) #Progress bar // Displays a progress bar while iterating through batches
            training_bar.set_description('Running Loss: %f' % (generator_running_loss/len(train_set)))
            generator_running_loss = 0.0
            
            for hr_img, lr_img in training_bar:
                if torch.cuda.is_available() and opt.cuda:
                    hr_img = hr_img.cuda()
                    lr_img = lr_img.cuda()
                sr_img = generator_net(lr_img)
                        
                content_loss = content_criterion(sr_img, hr_img)
                perceptual_loss = content_criterion(feature_extractor(sr_img), feature_extractor(hr_img))
                generator_loss = content_loss + 2e-8*tv_reg(sr_img) #  + 0.006*perceptual_loss
                
                generator_loss.backward()
                generator_optimizer.step()
                
                #Gets the numerical loss value from the loss tensor (removing unnecessary gradient tracking).
                #hr_img.size(0) Gets the batch size, meaning the number of images in this batch.
                generator_running_loss += generator_loss.item() * hr_img.size(0)
                generator_net.zero_grad() # clears previous gradients to prevent accumulation.

            torch.save(generator_net.state_dict(), weights_dir+ 'G_epoch_%d.pth' % (epoch)) #Saves model weights.
            generator_losses.append((epoch,generator_running_loss/len(train_set))) #Tracks epoch-wise loss.
                        
            #Every N epochs, validation images are saved, and PSNR (Peak Signal-to-Noise Ratio) is computed.
            if epoch % 1 ==0: 
                #Disables gradient calculation for evaluation 
                with torch.no_grad():
                    #Creates a directory to save generated images
                    cur_epoch_dir = imgout_dir+str(epoch)+'/'
                    os.makedirs(cur_epoch_dir, exist_ok=True)
                        
                    #Puts the generator into evaluation mode and prepares for validation.
                    generator_net.eval()
                    valid_bar = tqdm(validloader)
                    img_count = 0
                    psnr_avg = 0.0
                    psnr = 0.0
                        
                    for hr_img, lr_img in valid_bar:
                        valid_bar.set_description('Img: %i   PSNR: %f' % (img_count ,psnr))
                        
                        #Moves images to GPU.
                        if torch.cuda.is_available():
                            lr_img = lr_img.cuda()
                            hr_img = hr_img.cuda()
                        
                        #Calculates MSE (Mean Squared Error) between the generated SR image and the original HR image.
                        #Uses PSNR, a standard metric for measuring image quality (higher PSNR = better quality).
                        sr_tensor = generator_net(lr_img)
                        mse = torch.mean((hr_img-sr_tensor)**2)
                        psnr = 10* (torch.log10(1/mse) + np.log10(4))
                        psnr_avg += psnr
                        
                        #Keeps track of the number of images processed.
                        #Saves SR and LR images for visual inspection.
                        img_count +=1
                        sr_img = sr_transform(sr_tensor[0].data.cpu())
                        lr_img = lr_transform(lr_img[0].cpu())
                        sr_img.save(cur_epoch_dir+'sr_' + str(img_count)+'.png')
                        lr_img.save(cur_epoch_dir+'lr_'+str(img_count)+'.png')

                    #Averages the PSNR values across all validation images and stores them.
                    psnr_avg /= img_count
                    PSNR_valid.append((epoch, psnr_avg.cpu()))

                check_point = {'generator': generator_net.state_dict(), 'generator_optimizer': generator_optimizer.state_dict(), 'generator_losses': generator_losses ,'PSNR_valid': PSNR_valid}
                        
                torch.save(check_point, check_points_dir + 'check_point_epoch_%d.pth' % (epoch))	
                #Saves the losses and PSNR values as text files for analysis.
                np.savetxt(opt.out_dir + "generator_losses", generator_losses, fmt='%i,%f')
                np.savetxt(opt.out_dir + "PSNR", PSNR_valid, fmt='%i, %f')
                
                #save plot
                #Discriminator loss is not needed here, so we pass [].
                save_plot(generator_losses, discriminator_losses if opt.mode == "adversarial" else [], PSNR_valid, opt.mode)


#=========================================Adversarial training===============================================#
    if opt.mode == 'adversarial':
        #This will track the discriminator's total loss over an epoch.
        discriminator_running_loss = 0.0

        for epoch in range(start_epoch, opt.epochs+1):
            print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
            #This ensures both networks update their weights during training.
            generator_net.train()
            discriminator_net.train()
                        
            training_bar = tqdm(trainloader)
            training_bar.set_description('G: %f    D: %f' % (generator_running_loss/len(train_set), discriminator_running_loss/len(train_set)))
                        
            generator_running_loss = 0.0
            discriminator_running_loss = 0.0
                        
            for hr_img, lr_img in training_bar:
                #This prevents you from being overconfident, so if someone tries to slip in a super high-quality fake, you won’t just accept it blindly
                #hr_labels = torch.from_numpy(np.random.random((hr_img.size(0),1)) * 0.1 + 0.95).float()
                
                # Ensure labels are in the right shape (batch_size, 1)
                hr_labels = torch.ones(hr_img.size(0), 1).float()  # Fixed label shape for real images
                sr_labels = torch.zeros(hr_img.size(0), 1).float()  # Fixed label shape for fake images
                
                #generator wants the discriminator to think its fake images are real (1.0).
                ones = torch.ones(hr_img.size(0), 1).float()
                
                # for the discriminator, telling it that the generated (SR) images are fake.
                #creates a tensor of 0s (e.g., [[0.0], [0.0], [0.0], ...]).
                #sr_labels = torch.zeros(hr_img.size(0), 1).float()
                        
                if torch.cuda.is_available() and opt.cuda:
                    hr_img = hr_img.cuda()
                    lr_img = lr_img.cuda()
                    hr_labels = hr_labels.cuda()
                    sr_labels = sr_labels.cuda()
                    ones = ones.cuda()     
                sr_img = generator_net(lr_img)  #The generator takes the LR image and outputs a super-resolution image.
                                       
                generator_net.zero_grad()
                discriminator_net.zero_grad()
                        
                        
                #===================== train generator =====================
                #discriminator looks at the generated image (sr_img) and predicts whether it's real or fake
                #ones means we are tricking the discriminator into believing "Hey, this is a real image!"
                adversarial_loss = adversarial_criterion(discriminator_net(sr_img), ones)
                        
                #generated images "look good" beyond just pixel accuracy.
                perceptual_loss = content_criterion(feature_extractor(sr_img), feature_extractor(hr_img))
                content_loss = content_criterion(sr_img, hr_img) #mse
                generator_loss =  0.006*perceptual_loss + 1e-3*adversarial_loss  + content_loss 
                """
                `Content loss (most weight, ensuring images look correct).
                `Perceptual loss (helps maintain high-level details).
                `Adversarial loss (ensures the discriminator gets tricked).
                """
                
                generator_loss.backward()
                generator_optimizer.step()
                #===================== train discriminator =====================
                #calculates how well the discriminator is performing by balancing two losses
                discriminator_loss = (adversarial_criterion(discriminator_net(hr_img), hr_labels) + \
                                    adversarial_criterion(discriminator_net(sr_img.detach()), sr_labels))/2
                
                discriminator_loss.backward()
                discriminator_optimizer.step()
                generator_running_loss += generator_loss.item() * hr_img.size(0)
                discriminator_running_loss += discriminator_loss.item() * hr_img.size(0)


            torch.save(generator_net.state_dict(), weights_dir+ 'G_epoch_%d.pth' % (epoch))
            generator_losses.append((epoch,generator_running_loss/len(train_set)))
            discriminator_losses.append((epoch,discriminator_running_loss/len(train_set)))
                        
            if epoch % 1 ==0:
                with torch.no_grad():
                    cur_epoch_dir = imgout_dir+str(epoch)+'/'
                    os.makedirs(cur_epoch_dir, exist_ok=True)
                    generator_net.eval()
                    discriminator_net.eval()
                    valid_bar = tqdm(validloader)
                    img_count = 0
                    psnr_avg = 0.0
                    psnr = 0.0
                        
                    for hr_img, lr_img in valid_bar:
                        valid_bar.set_description('Img: %i   PSNR: %f' % (img_count ,psnr))
                        if torch.cuda.is_available():
                            lr_img = lr_img.cuda()
                            hr_img = hr_img.cuda()
                        sr_tensor = generator_net(lr_img)
                        mse = torch.mean((hr_img-sr_tensor)**2)
                        psnr = 10* (torch.log10(1/mse) + np.log10(4))
                        psnr_avg += psnr
                        img_count +=1
                        sr_img = sr_transform(sr_tensor[0].data.cpu())
                        lr_img = lr_transform(lr_img[0].cpu())
                        sr_img.save(cur_epoch_dir+'sr_' + str(img_count)+'.png')
                        lr_img.save(cur_epoch_dir+'lr_'+str(img_count)+'.png')


                    psnr_avg /= img_count
                    PSNR_valid.append((epoch, psnr_avg.cpu()))

                check_point = {'generator': generator_net.state_dict(), 'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator': discriminator_net.state_dict(), 'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'discriminator_losses': discriminator_losses, 'generator_losses': generator_losses ,'PSNR_valid': PSNR_valid}
                torch.save(check_point, check_points_dir + 'check_point_epoch_%d.pth' % (epoch))	
                np.savetxt(opt.out_dir + "generator_losses", generator_losses, fmt='%i,%f')
                np.savetxt(opt.out_dir + "discriminator_losses", discriminator_losses, fmt='%i, %f')
                np.savetxt(opt.out_dir + "PSNR", PSNR_valid, fmt='%i, %f')
                        
                #Save plot
                save_plot(generator_losses, discriminator_losses if opt.mode == "adversarial" else [], PSNR_valid, opt.mode)













