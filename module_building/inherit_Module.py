import torch
from torch import nn

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。
        # 这样在构造实例时还可以指定其他函数参数
        # 如“模型参数的访问、初始化和共享”一节介绍的模型参数param
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10) # 输出层
    
    # 定义模型的前向计算，即如何根据输入x计算返回所需的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

x = torch.rand(2, 784)
# net = MLP()
# print(net)
# print(net(x))

class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # 如果传入的是一个OrderdDict
            for key, module in args[0].items():
                # add_module方法会将module添加进self._modules(一个OrderdDict)
                self.add_module(key, module)
        else: # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    def forward(self, input):
        # self._modules返回一个OrderdDict,保证会按照成员添加时的顺序遍历
        for module in self._modules.values():
            input = module(input)
        return input

# 用MySeqyebtuak类来实现前面描述的MLP类
# net = MySequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )
# print(net)
# print(net(x))

# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10)) # 类似List的append操作
# print(net[-1]) # 类似List的索引访问
# print(net)

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256), 
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)