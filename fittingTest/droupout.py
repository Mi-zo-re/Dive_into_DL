import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

def dropout(x, drop_prob):
    x = x.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(x)
    mask = (torch.randn(x.shape) < keep_prob).float()

    return mask * x / keep_prob

# x = torch.arange(16).view(2, 8)
# print(dropout(x, 0))
# print(dropout(x, 0.5))
# print(dropout(x, 1.0))
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
w1 = torch.tensor(np.random.normal(0, 0.01, size = (num_inputs, num_hiddens1)), dtype = torch.float, requires_grad = True)
b1 = torch.zeros(num_hiddens1, requires_grad = True)
w2 = torch.tensor(np.random.normal(0, 0.01, size = (num_hiddens1, num_hiddens2)), dtype = torch.float, requires_grad = True)
b2 = torch.zeros(num_hiddens2, requires_grad = True)
w3 = torch.tensor(np.random.normal(0, 0.01, size = (num_hiddens2, num_outputs)), dtype = torch.float, requires_grad = True)
b3 = torch.zeros(num_outputs, requires_grad = True)

params = [w1, b1, w2, b2, w3, b3]

drop_prob1, drop_prob2 = 0.2, 0.5

# def net(x, is_training = True):
#     x = x.view(-1, num_inputs)
#     H1 = (torch.matmul(x, w1) + b1).relu()
#     if is_training: # 只在训练模型时使用丢弃法
#         H1 = dropout(H1, drop_prob1) # 在第一层全连接后添加丢弃层
#     H2 = (torch.matmul(H1, w2) + b2).relu()
#     if is_training:
#         H2 = dropout(H2, drop_prob2) # 在第二层全连接后添加丢弃层
#     return torch.matmul(H2, w3) + b3

num_epochs, lr, batch_size =5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
# 简洁实现
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, 10)
)

for param in net.parameters():
    nn.init.normal_(param, mean = 0, std = 0.01)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
d2l.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
