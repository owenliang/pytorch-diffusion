import torch 
from config import *
from dataset import train_dataset

# 前向diffusion计算参数
betas=torch.linspace(0.0001,0.02,T) # (T,)
alphas=1-betas  # (T,)

alphas_cumprod=torch.cumprod(alphas,dim=-1) # alpha_t累乘 (T,)
alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,)
variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)  # denoise用的方差   (T,)

# 执行前向加噪
def forward_diffusion(batch_x,batch_t): # batch_x: (batch,channel,width,height), batch_t: (batch_size,)
    batch_noise_t=torch.randn_like(batch_x).to(DEVICE)   # 为每张图片生成第t步的高斯噪音   (batch,channel,width,height)
    batch_alphas_cumprod=alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0),1,1,1) 
    batch_x_t=torch.sqrt(batch_alphas_cumprod)*batch_x+torch.sqrt(1-batch_alphas_cumprod)*batch_noise_t # 基于公式直接生成第t步加噪后图片
    return batch_x_t,batch_noise_t

if __name__=='__main__':
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) # 2个图片拼batch, (2,1,96,96)
    batch_x=batch_x*2-1 # 像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  # 每张图片随机生成diffusion步数
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)
    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())