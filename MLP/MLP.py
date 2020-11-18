import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定于模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype = torch.float)
b1 = torch.zeros(num_hiddens, dtype = torch.float)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype = torch.float)
b2 = torch.zeros(num_outputs, dtype = torch.float)

params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad = True)

# 定义激活函数ReLU
def relu(x):
    return torch.max(input = x, other = torch.tensor(0.0))

# 定义模型
def net(x):
    x = x.view((-1, num_inputs))
    h = relu(torch.matmul(x, w1) + b1)
    return torch.matmul(h, w2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

#训练模型
num_epochs, lr = 5, 100.0
d2l.train_softmax(net, train_iter, test_iter,loss, num_epochs, batch_size, params, lr)
