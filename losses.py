import torch
import torch.nn as nn
from torchvision import transforms

"""
MeanShift adjusts the pixel values of an image so that they match the format the VGG model is designed to work with. It doesn't affect how the model learns or how the gradients (used for training) behave; it just makes sure the image data is in the right range or scale for the model to process effectively.
"""
class MeanShift(nn.Conv2d):
    #rgb_range: pixel value range (e.g., 1.0 if normalized to [0,1], or 255 if not).
    #norm_mean and norm_std: the mean and std values used for normalizing images (same as in ImageNet and VGG).
    #-1 for normalization (subtract mean, divide std)
    #+1 for denormalization (multiply std, add mean)
    def __init__(
        self, rgb_range = 1,
        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(norm_std)  #Converts the std tuple to a PyTorch tensor.
        #Hey, don’t learn anything fancy. Just pass the image through unchanged — but first, divide each color channel (Red, Green, Blue) by its standard deviation.
        #Take the image and divide the Red channel by Red’s std, Green by Green’s std, Blue by Blue’s std
        #eye creates a 3 x 3 matrix
        #.view reshape Format: [output_channels, input_channels, height, width]
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(norm_mean) / std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Freezes the weights and biases — this layer is not trainable.
        #It’s just a fixed normalization layer, not something the model should learn.
        for p in self.parameters():
            p.requires_grad = False
            
            
class perceptual_loss(nn.Module):
    def __init__(self, vgg):
        super(perceptual_loss, self).__init__()
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MeanShift(norm_mean = self.normalization_mean, norm_std = self.normalization_std).to(self.device)
        self.vgg = vgg
        self.criterion = nn.MSELoss()  #It’s used to measure how different the deep features are between HR and SR.
    def forward(self, HR, SR, layer = 'relu5_4'):
        ## HR and SR should be normalized [0,1]
        hr = self.transform(HR)
        sr = self.transform(SR)
        
        #Extracts intermediate feature maps from VGG (like relu5_4) for both images.
        #getattr() accesses that layer by name (your VGG model must return feature maps in a dict or similar).
        hr_feat = getattr(self.vgg(hr), layer)
        sr_feat = getattr(self.vgg(sr), layer)
        
        return self.criterion(hr_feat, sr_feat), hr_feat, sr_feat
    #The feature maps for HR and SR (optional — could be used for analysis or debugging).
    #The perceptual loss (MSE between the VGG feature maps).

class TVLoss(nn.Module):
    #tv_loss_weight lets you control how much you want to penalize image roughness.
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    #x is usually the output image (SR image) from your model.
    def forward(self, x):  
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        #Counts how many pixels will be compared vertically and horizontally.
        count_h = self.tensor_size(x[:, :, 1:, :]) #basically if it is matrix, it counts the num of elements
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3] #C×H×W
    
    
    
    