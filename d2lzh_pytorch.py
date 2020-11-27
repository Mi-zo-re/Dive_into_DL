from IPython import display
from matplotlib import pyplot as plt
import random
import torch
import sys
import torchvision
import torchvision.transforms as transforms
from torch import nn
import time
import torchvision

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size, resize=None, root='E:/DLonPyTorch/datasets/FashionMNIST'):
    # mnist_train = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = True, download = True, transform = transforms.ToTensor())
    # mnist_test = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = False, download = True, transform = transforms.ToTensor())

    # mnist_train = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = True, download = False, transform = transforms.ToTensor())
    # mnist_test = torchvision.datasets.FashionMNIST(root = 'E:/DLonPyTorch/datasets/FashionMNIST', train = False, download = False, transform = transforms.ToTensor())


    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    # train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    # test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    # 修改版
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, 
        train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
        train=False, download=False, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, 
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, 
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (train_iter, test_iter)

def evaluate_accuracy(data_iter, net,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式，这会关闭dropout
                acc_sum += (net(x.to(device)).argmax(dim = 1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 之后用不到所以不考虑GPU
                if ('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(x, is_training = False).argmax(dim = 1) == y).float().sum().item()
                else:
                    acc_sum += (net(x).argmax(dim = 1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr = None, optimizer = None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals = None, y2_vals = None, legend = None, figsize = (3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle = ":")
        plt.legend(legend)
    plt.show()

def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, 
        x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i + h, j: j + w] * k).sum()
    return y

def train_conv(net, train_iter, test_iter, 
        batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))