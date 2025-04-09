import numpy as np
import torch
import torch.nn as nn


def upsample_block(in_channels):
    #This block doubles the image size, which is needed to create a high-resolution image.
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels*4, 3, padding=1), #
        nn.PixelShuffle(2), #Rearranges the channels to double the image resolution (e.g., from 64×64 → 128×128)
        nn.PReLU(in_channels)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x #Skip connection [Adds the original input (x) back to the output.] // This helps prevent the model from forgetting details.
        return out

class Generator(nn.Module):
    #This generator takes a low-resolution image and creates a high-resolution version using residual learning and upsampling.
    def __init__(self, upscale_factor=4, num_blocks=16):
        super().__init__()
        num_upblocks = int(np.log2(upscale_factor)) #calculation for  determines the number of upsampling blocks required to upscale an image by a given upscale_factor
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.relu = nn.PReLU(64)
        self.resblocks = nn.Sequential(*([ResidualBlock(64, 64)]* num_blocks))
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.upblocks = nn.Sequential(*([upsample_block(64)]* num_upblocks))
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        identity = self.relu(out)  # Save the original image features
        out = self.resblocks(identity)    # Pass through Residual Blocks
        out = self.conv2(out)
        out = self.bn(out)
        out += identity      # Skip connection for stability
        out = self.upblocks(out)  # Upsample to higher resolution
        out = self.conv3(out)
        return torch.tanh(out)  # Normalize output between -1 and 1
    

#Discriminator
def convolution_block(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
        )

class Discriminator(nn.Module):
    def __init__(self, crop_size = 128):
        super().__init__()
        num_ds = 4
        size_list = [64, 128, 128, 256, 256, 512, 512] #Defines the number of channels at each layer.
        stride_list = [1,2]*3  # [1,2,1,2,1,2]
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        #The * expands the list so that each element is passed as a separate argument to nn.Sequential().
        #The last value in size_list is ignored because zip() stops at the shortest iterable.
        self.convblocks = nn.Sequential(convolution_block(64, 64, 2), 
                                        *[convolution_block(in_ch, out_ch, stride)
                                        for in_ch, out_ch, stride in zip(size_list,size_list[1:],stride_list)])
        # Converts spatial features into a 1D vector before passing it to the classifier.
        self.fc1 = nn.Linear(int(512*(crop_size/ 2**num_ds)**2), 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()  #Outputs a probability between 0 and 1.

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)  # Activation
        out = self.convblocks(out) # Feature extraction (batch_size, num_channels, height, width)
        
        # Debugging: print the shape before passing to fully connected layer
        #print(out.shape)  # This will print the shape of the output before flattening
        
        #out.size(0): This keeps the batch size unchanged.
        #-1: This automatically infers the remaining dimensions and flattens the feature map into a single vector per sample. [(batch_size, 512 * 8 * 8) = (batch_size, 32768)]
        out = self.fc1(out.view(out.size(0),-1)) # Flatten & FC1
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out
    
"""  
Step	                        Shape Transformation
Input Image	                    (batch_size, 3, 128, 128)
After Conv Layers (convblocks)	(batch_size, 512, 8, 8)
After Flattening (view)	        (batch_size, 32768)
After FC Layer (fc1)	        (batch_size, 1024)
"""
