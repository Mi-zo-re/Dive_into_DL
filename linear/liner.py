# 线性回归测试
# %matplotlib inline
import torch
import numpy as np
import random
import sys
sys.path.append("..")
from d2lzh_pytorch import *

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

# print(features[0], labels[0])

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float64)
b = torch.zeros(1, dtype=torch.float64)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

lr = 0.03
num_epochs = 5
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)