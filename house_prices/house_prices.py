import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

# print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('../datasets/kaggle_house/train.csv')
test_data = pd.read_csv('../datasets/kaggle_house/test.csv')

# 1460个sample，80个feature，1个label
# print(train_data.shape)
# 1459个sample，80个feature，预测价格
# print(test_data.shape)

# print(train_data.iloc[0: 4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))
# 特征标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 标准化后，没个特征值的平均值变为0，所以可以直接用0来替换缺失值
all_features = all_features.fillna(0)

# 将原特征里的离散值转换为独立的特征，用0、1值表示
# dummy_na = True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na = True)
# print(all_features.shape)# (2919, 354)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype = torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype = torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype = torch.float).view(-1, 1)

# 训练模型
loss = torch.nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean = 0, std = 0.01)
    return net

# 比赛用来评价模型的对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设为1， 使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params = net.parameters(),lr = learning_rate, weight_decay = weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, x, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim = 0)
            y_train = torch.cat((y_train, y_part), dim = 0)
    return x_train, y_train, x_valid, y_valid

# 训练k次并返回训练和验证的平均误差
def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net(x_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

# 预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis = 1)
    submission.to_csv('./submission.csv', index = False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)