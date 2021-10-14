import torch
import numpy as np
import random
import torchvision.transforms.functional as TF

class RandomRotation(object):
    def __init__(self, choices=4, include_zero=True):
        angles = np.linspace(0,360,choices+1)
        angles = angles[:-1]
        
        if not include_zero:
            delta = (360-angles[-1])/2
            angles = angles+delta
            
        self.choices = angles
        
    def __call__(self, tensor):
        angle = random.choice(self.choices)
        return TF.rotate(tensor, angle)
    

class GaussianNoise(object):
    def __init__(self, sigma=0.1, channels=[]):
        self.sigma = sigma
        self.channels = channels
        
        
    def __call__(self, tensor):
        if self.sigma != 0:
            scale = self.sigma * tensor
            sampled_noise = torch.tensor(0,dtype=torch.float32).repeat(*tensor.size()).normal_() * scale
            
            # remove noise for masked channels
        
            for channel in range(tensor.shape[0]):
                if channel not in self.channels:
                    sampled_noise[channel,:,:] = 0
        
            tensor = tensor + sampled_noise
        return tensor
    
class ChannelReducer(object) :
    def __init__(self, channels=5):
        """
        can reduce a golgi dataset to 5,3 or 1 channel
        5: nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_cellmask
        3: nuclei_mask,cell_mask,channel_golgi
        1: channel_golgi
     """        
        self.channels=channels
        
    def __call__(self, tensor):
        
        if self.channels == 1:
            return tensor [[3],:,:]
        elif self.channels == 3:
            return tensor [[0,1,3],:,:]
        else:
            return tensor
            
class ChannelSelector(object) :
    def __init__(self, channels=[0,1,2,3,4], num_channels=5):
        """
        select the channels used for prediction
     """        
        
        if not np.max(channels) < num_channels:
            raise ValueError("highest channel index exceeds channel numb")
        self.channels=channels
        
    def __call__(self, tensor):
        return tensor[self.channels,:,:]
         