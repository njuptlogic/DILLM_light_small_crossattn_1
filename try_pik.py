'''
import pickle

# 设置解压后的文件路径
file_path = '/home/ubuntu/Documents/DILLM/best_val_unseen'

# 加载并查看 pickle 文件内容
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)  # 打印内容查看
except Exception as e:
    print(f"Error loading the file: {e}")
'''

import torch

# 使用 PyTorch 的 load 方法来加载模型
try:
    data = torch.load('/home/ubuntu/Documents/DILLM/best_val_unseen')
    print(data)  # 打印内容查看
except Exception as e:
    print(f"Error loading the file: {e}")
