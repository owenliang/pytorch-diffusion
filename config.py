import torch 

IMG_SIZE=48   # 图像尺寸
T=1000   # 加噪最大步数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备