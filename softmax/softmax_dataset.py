import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# mnist_train = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = True, download = True, transform = transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = False, download = True, transform = transforms.ToTensor())

# print(type(mnist_test))
# print(len(mnist_train), len(mnist_test))

# feature, label = mnist_train[0]
# print(feature.shape, label)
# 1维是通道数，2维3维分别是图片的高和宽

# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

batch_size = 256
# 已写入d2lzh_pytorch中
# if sys.platform.startswith('win'):
#     num_workers = 0 # 0 表示不用额外的进程来加速读取数据
# else:
#     num_workers = 4
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
# test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = num_workers)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

start = time.time()
# 读取一遍训练集所需要的时间
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))