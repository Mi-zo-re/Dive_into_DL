import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 函数图像
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()

def grad_plot(x_vals, y_vals, name):
    if x.grad is not None:
        x_vals.grad.zero_()
    y_vals.sum().backward()
    xyplot(x, x.grad, 'grad of ' + name)

# ReLU函数
name = 'relu'
x = torch.arange(-8.0, 8.0, 0.1, requires_grad = True)
y = x.relu()
xyplot(x, y, name)
grad_plot(x, y, name)

# Sigmoid函数
name = 'sigmoid'
y = x.sigmoid()
xyplot(x, y, name)
grad_plot(x, y, name)

# tanh函数
name = 'tanh'
y = x.tanh()
xyplot(x, y, name)
grad_plot(x, y, name)