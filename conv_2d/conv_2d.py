import torch
from torch import nn
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias

x = torch.ones(6, 8)
x[:, 2: 6] = 0
# print(x)
k = torch.tensor([[1, -1]])
# print(k)
y = d2l.corr2d(x, k)
# print(y)

conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    y_hat = conv2d(x)
    l = ((y_hat - y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    
    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('step %d, loss %.3f' % (i + 1, l.item()))
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)