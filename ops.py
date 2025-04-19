import torch
import torch.nn as nn
import torch.nn.functional as F

#_xxx only for internal use of this functio0n
class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        # it uses torch.normal to give the filter values that are randomly generated from a normal distribution with a mean of 0 and a standard deviation of 0.02. This helps in starting the training with random weights.
        #help system start better 
        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))
        
        #This ensures that the parameters (weights and bias) can be updated during training.
        for p in self.parameters():
            p.requires_grad = True
        

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        #BN (Batch Normalization): A flag that, if set to True, adds a batch normalization layer after the convolution. This layer helps stabilize and speed up training
        if BN:
            m.append(nn.BatchNorm2d(num_features = out_channel))
        
        #act: The activation function to use (like ReLU), which adds non-linearity to the layer. If None, no activation function is applied.
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act = nn.ReLU(inplace = True)):
        super(BasicBlock, self).__init__()
        m = []
        
        self.conv = conv(in_channels, out_channels, kernel_size, BN = False, act = act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))
        
        m.append(conv(out_channels, out_channels, kernel_size, BN = True, act = None))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res
        
        return out
        
class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        #This calls the constructor of the parent class nn.Module. Always needed when making custom layers.
        super(Upsampler, self).__init__()
        m = []
        #Go see notes.ipyb
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale)) #For example, if scale = 2, it turns a 16×16 image into a 32×32 image by smartly rearranging the extra channels into pixels.
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act = nn.LeakyReLU(inplace = True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN = True, act = act))
        m.append(conv(out_feats, out_feats, kernel_size, BN = True, act = act, stride = 2))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out