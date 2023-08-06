from unet import UNet
from dataset import train_dataset
from diffusion import forward_diffusion
from config import * 
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
from lora import inject_lora

EPOCH=200
BATCH_SIZE=400

if __name__=='__main__':
    # 模型
    model=torch.load('model.pt')

    # 向nn.Linear层注入Lora
    for name,layer in model.named_modules():
        name_cols=name.split('.')
        # 过滤出cross attention使用的linear权重
        filter_names=['w_q','w_k','w_v']
        if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
            inject_lora(model,name,layer)
    
    # lora权重的加载
    try:
        restore_lora_state=torch.load('lora.pt')
        model.load_state_dict(restore_lora_state,strict=False)
    except:
        pass 

    model=model.to(DEVICE)

    # 冻结非Lora参数
    for name,param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a','lora_b']:  # 非LOra部分不计算梯度
            param.requires_grad=False
        else:
            param.requires_grad=True

    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=4,persistent_workers=True,shuffle=True)   # 数据加载器

    optimizer=torch.optim.Adam(filter(lambda x: x.requires_grad==True,model.parameters()),lr=0.001) # 优化器只更新Lorac参数
    loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

    print(model)

    writer = SummaryWriter()
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
            optimizer.step()
            last_loss=loss.item()
            writer.add_scalar('Loss/train', last_loss, n_iter)
            n_iter+=1
        print('epoch:{} loss={}'.format(epoch,last_loss))

        # 保存训练好的Lora权重
        lora_state={}
        for name,param in model.named_parameters():
            name_cols=name.split('.')
            filter_names=['lora_a','lora_b']
            if any(n==name_cols[-1] for n in filter_names):
                lora_state[name]=param
        torch.save(lora_state,'lora.pt.tmp')
        os.replace('lora.pt.tmp','lora.pt')