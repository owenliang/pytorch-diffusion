import torch 

IMG_SIZE=96   # 图像尺寸
BATCH_SIZE=256    # 训练批次大小
T=300   # 加噪最大步数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备