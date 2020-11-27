import torch
from torch import nn

def pool2d(x, pool_size, mode='max'):
    x = x.float()
    p_h, p_w = pool_size
    y = torch.zeros(x.shape[0] -p_h + 1,
        x.shape[1] - p_w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode == 'max':
                y[i, j] = x[i: i + p_h, j: j+ p_w].max()
            elif mode == 'avg':
                y[i, j] = x[i: i + p_h, j: j + p_w].mean()
    return y

x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7,8]])
# print(pool2d(x, (2, 2)))
# print(pool2d(x, (2, 2), 'avg'))

x = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
# print(x)
pool2d = nn.MaxPool2d(3)
# print(pool2d(x))
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# print(pool2d(x))
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
# print(pool2d(x))

x = torch.cat((x, x + 1), dim=1)
# print(x)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(x))