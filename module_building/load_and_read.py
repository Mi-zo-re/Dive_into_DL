import torch
from torch import nn

# 存取张量
x = torch.ones(3)
# torch.save(x, 'x.pt')
# x2 = torch.load('x.pt')
# print(x)

# 存取张量列表
y = torch.zeros(4)
# torch.save([x, y], 'xy.pt')
# xy_list = torch.load('xy.pt')
# print(xy_list)

# 存取张量字典
# torch.save({'x':x, 'y':y}, 'xy_dict.pt')
# xy = torch.load('xy_dict.pt')
# print(xy)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
# print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print(optimizer.state_dict())

x = torch.randn(2, 3)
y = net(x)
PATH = './net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
y2 = net2(x)
print(y2 == y)