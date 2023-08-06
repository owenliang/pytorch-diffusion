from config import *
from torch.utils.data import DataLoader
from dataset import train_dataset
from unet import UNet
from diffusion import forward_diffusion
import torch 
from torch import nn 
import os 
from torch.utils.tensorboard import SummaryWriter

EPOCH=200
BATCH_SIZE=400

dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=4,persistent_workers=True,shuffle=True)   # 数据加载器

try:
    model=torch.load('model.pt')
except:
    model=UNet(1).to(DEVICE)   # 噪音预测模型

optimizer=torch.optim.Adam(model.parameters(),lr=0.001) # 优化器
loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

writer = SummaryWriter()

if __name__=='__main__':
    model.train()
    n_iter=0
    for epoch in range(EPOCH):
        last_loss=0
        for batch_x,batch_cls in dataloader:
            # 图像的像素范围转换到[-1,1],和高斯分布对应
            batch_x=batch_x.to(DEVICE)*2-1
            # 引导分类ID
            batch_cls=batch_cls.to(DEVICE)
            # 为每张图片生成随机t时刻
            batch_t=torch.randint(0,T,(batch_x.size(0),)).to(DEVICE)
            # 生成t时刻的加噪图片和对应噪音
            batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)
            # 模型预测t时刻的噪音
            batch_predict_t=model(batch_x_t,batch_t,batch_cls)
            # 求损失
            loss=loss_fn(batch_predict_t,batch_noise_t)
            # 优化参数
            optimizer.zero_grad()
            loss.backward()
            # for name,param in model.named_parameters():
            #     name_cols=name.split('.')
            #     if name=='dec_convs.0.crossattn.w_q.weight':
            #         print(name,param.grad)
            optimizer.step()
            last_loss=loss.item()
            writer.add_scalar('Loss/train', last_loss, n_iter)
            n_iter+=1

        print('epoch:{} loss={}'.format(epoch,last_loss))
        torch.save(model,'model.pt.tmp')
        os.replace('model.pt.tmp','model.pt')