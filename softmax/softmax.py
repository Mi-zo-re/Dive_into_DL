import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float)
b = torch.zeros(num_outputs, dtype = torch.float)

w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

# dim即只对矩阵中的同一行或者同一列做加法运算，并保留行和列这两个维度
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x.sum(dim = 0, keepdim = True))
# print(x.sum(dim = 1, keepdim = True))

# 实现softmax运算
def softmax(X):
    X_exp = X.exp() # 指数运算
    partition = X_exp.sum(dim = 1, keepdim = True) # 每行元素求和
    return X_exp / partition # 各元素与该行元素和的商

# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim = 1))

# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), w) + b)

# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
# print(y_hat.gather(1, y.view(-1, 1)))
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim = 1) == y).float().mean().item()
print(accuracy(y_hat, y))
print(d2l.evaluate_accuracy(test_iter, net))

# 训练模型
num_epochs, lr = 5, 0.1
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
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
train_softmax(net, train_iter, test_iter, cross_entropy,num_epochs, batch_size, [w, b], lr)

# 预测
x, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim = 1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(x[0: 9], titles[0: 9])
