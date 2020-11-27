import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, x):
    # (1, 1)代表批量大小和通道数均为1
    x = x.view((1, 1) + x.shape)
    y = conv2d(x)
    return y.view(y.shape[2:]) # 排除不关心的前两维：批量和通道

# 注一这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, 
    kernel_size=3, padding=1)

x = torch.rand(8, 8)
# print(comp_conv2d(conv2d, x).shape)

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
    kernel_size=(5, 3), padding=(2, 1))
# print(comp_conv2d(conv2d, x).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3,
    padding=1, stride=2)
# print(comp_conv2d(conv2d, x).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5),
    padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, x).shape)