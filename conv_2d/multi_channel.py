import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def corr2d_multi_in(x, k):
    # 沿着x和k的第0维（通道维）分别计算再相加
    res = d2l.corr2d(x[0, :, :], k[0, :, :])
    for i in range(1, x.shape[0]):
        res += d2l.corr2d(x[i, :, :], k[i, :, :])
    return res

x = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
k = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
# print(corr2d_multi_in(x, k))

def corr2d_multi_in_out(x, k):
    # 对k的第0维遍历，每次同输入x做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(x, i) for i in k])

k = torch.stack([k, k + 1, k + 2])
# print(k.shape)
# print(k)

# print(corr2d_multi_in_out(x, k))

def corr2d_multi_in_out_1x1(x, k):
    c_i, h, w = x.shape
    c_o = k.shape[0]
    x = x.view(c_i, h * w)
    k = k.view(c_o, c_i)
    y = torch.mm(k, x) # 全连接层的矩阵乘法
    return y.view(c_o, h, w)

x = torch.rand(3, 3, 3)
k = torch.rand(2, 3, 1, 1)
y1 = corr2d_multi_in_out_1x1(x, k)
y2 = corr2d_multi_in_out(x, k)
print((y1 - y2).norm().item() < 1e-6)