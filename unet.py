import torch 
from torch import nn 
from dataset import train_dataset

class UNet(nn.Module):
    def __init__(self,img_channel):
        super().__init__()
    
    def forward(self,x):
        return 

if __name__=='__main__':
    img=train_dataset[0][0]
    batch_x=img.unsqueeze(0)