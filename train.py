from config import *
from torch.utils.data import DataLoader
from dataset import train_dataset
from unet import UNet
from diffusion import forward_diffusion
import torch 
from torch import nn 
import os 

EPOCH=10
BATCH_SIZE=400

dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=4,persistent_workers=True,shuffle=True)   # 数据加载器

try:
    model=torch.load('model.pt')
except:
    model=UNet(1).to(DEVICE)   # 噪音预测模型

optimizer=torch.optim.Adam(model.parameters(),lr=0.001) # 优化器
loss_fn=nn.L1Loss()

if __name__=='__main__':
    for epoch in range(EPOCH):
        last_loss=0
        for batch_x,_ in dataloader:
            # 为每张图片生成随机t时刻
            batch_t=torch.randint(0,T,(batch_x.size(0),))
            # 生成t时刻的加噪图片和对应噪音
            batch_x_t,batch_noise_t=forward_diffusion(batch_x.to(DEVICE),batch_t.to(DEVICE))
            # 模型预测t时刻的噪音
            batch_predict_t=model(batch_x_t.to(DEVICE),batch_t.to(DEVICE))
            # 求损失
            loss=loss_fn(batch_predict_t,batch_noise_t)
            # 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss=loss.item()
        print('epoch:{} loss={}'.format(epoch,last_loss))
        torch.save(model,'model.pt.tmp')
        os.replace('model.pt.tmp','model.pt')