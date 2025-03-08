from __future__ import print_function
from distutils.version import LooseVersion
from matplotlib.scale import LogisticTransform
import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings(action='once')


class MaxMinLogRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)


class MaxMinRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        # data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)


class MaxMinLogScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data[data != 0] = -np.log(data[data != 0] + 1)
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data[data != 0] = (data[data != 0] - min) / (max - min + 1e-12)
        return data

    def inverse_transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data = data * (max - min) + min
        return np.exp(data)


class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


def graph_normal(graphs, normal=2):
    if normal == 2:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)

    elif normal == 1:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = [1, 1], [0, 0]

    return graphs


def get_data_generator(opt, data_train, data_test, normal=2):
    # load data
    data_train = data_train[np.array([graph["exist"] for graph in data_train])]
    data_test = data_test[np.array([graph["exist"] for graph in data_test])]

    data_train, data_test = graph_normal(data_train, normal=normal), graph_normal(data_test, normal=normal)

    random.seed(opt.seed)
    random.shuffle(data_train)
    random.seed(opt.seed)
    random.shuffle(data_test)

    return data_train, data_test

def square_sum(gamma1,gamma):
    out = ((gamma1-gamma)**2).sum(dim=1, keepdim=True) 
    return out 

# fusion two NIG
def fuse_nig(gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
    # Eq. 16
    gamma = (gamma1*v1 + gamma2*v2) / (v1+v2 + 1e-12)
    v = v1 + v2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (v1 * square_sum(gamma1, gamma) + v2 * square_sum(gamma2, gamma))
    return gamma, v, alpha, beta

def dis_loss(y, y_pred, max, min):
    y[:, 0] = y[:, 0] * (max[0] - min[0])
    y[:, 1] = y[:, 1] * (max[1] - min[1])
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0])
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1])
    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100)).sum(dim=1))
    return distance
#Haversine公式，考虑了地球的曲率
# def haversine_disloss(y, y_pred):
#     """
#     计算两点间的Haversine距离（单位：公里）
#     y, y_pred: 形状为[N, 2]的张量，每行是[经度, 纬度]
#     """
#     # 将经纬度转换为弧度
#     lat1 = torch.deg2rad(y[:, 1])
#     lon1 = torch.deg2rad(y[:, 0])
#     lat2 = torch.deg2rad(y_pred[:, 1])
#     lon2 = torch.deg2rad(y_pred[:, 0])
    
#     # Haversine公式
#     R = 6371  # 地球半径（公里）
    
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
    
#     a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
#     c = 2 * torch.asin(torch.sqrt(a))
    
#     return R * c

def NIG_NLL(gamma, v, alpha, beta, mse):
    om = 2 * beta * (1 + v)
    nll =  0.5*torch.log(np.pi/v + 1e-12) \
        - alpha*torch.log(om) \
        + (alpha + 0.5) * torch.log(v * mse + om) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    return torch.mean(nll)

def NIG_Reg(v, alpha, mse):
    reg = mse * (2 * v + alpha)
    return torch.mean(reg)

def NIG_loss(gamma, v, alpha, beta, mse, coeffi = 0.01):
    # our loss function
    om = 2 * beta * (1 + v)
    loss = \
        (0.5 * torch.log(np.pi / v + 1e-12) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(v * mse + om) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)).sum() / len(gamma)
    lossr = coeffi * (mse * (2 * v + alpha)).sum() / len(gamma)
    loss = loss + lossr
    return loss + lossr


def get_adjancy(func, delay, hop, nodes):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    hops = []
    delays = []
    x1 = []
    x2 = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            delays.append(delay[i, j])
            hops.append(hop[i, j])
            x1.append(nodes[i].cpu().detach().numpy())
            x2.append(nodes[j].cpu().detach().numpy())
    dis = func(Tensor(delays), Tensor(hops), Tensor(x1), Tensor(x2))
    A = torch.zeros_like(delay)
    index = 0
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            A[i, j] = dis[index]
            index += 1
    return A


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

# 初始化网络权重

def init_network_weights(net, std=0.1):
    #
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            # nn.init.constant_(m.bias, val=0)

# save checkpoint of model
def save_cpt(model, optim, epoch, save_path):
    """
    save checkpoint, for inference/re-training
    :return:
    """
    model.eval()
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
        save_path
    )

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))


def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))


def draw_cdf(ds_sort):
    last, index = min(ds_sort), 0
    x = []
    y = []
    while index < len(ds_sort):
        x.append([last, ds_sort[index]])
        y.append([index / len(ds_sort), index / len(ds_sort)])

        if index < len(ds_sort):
            last = ds_sort[index]
        index += 1
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(x).reshape(-1, 1).squeeze(),
             np.array(y).reshape(-1, 1).squeeze(),
             c='k',
             lw=2,
             ls='-')
    plt.xlabel('Geolocation Error(km)')
    plt.ylabel('Cumulative Probability')
    plt.grid()
    plt.show()

def kl_divergence(p, q):
    """
    计算KL散度: KL(p||q)
    p, q: 概率分布
    """
    return torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12) + 1e-12))

def adversarial_loss(clean_output, adv_output):
    """
    计算对抗鲁棒性损失
    """
    p_clean = F.softmax(clean_output, dim=1)
    p_adv = F.softmax(adv_output, dim=1)
    return kl_divergence(p_clean, p_adv)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
        
    Returns:
        distance: Distance between the points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def haversine_loss(pred, target):
    """
    Calculate Haversine loss between predicted and target coordinates
    
    Args:
        pred: Predicted coordinates [batch_size, 2] (lon, lat)
        target: Target coordinates [batch_size, 2] (lon, lat)
        
    Returns:
        loss: Mean Haversine distance in kilometers
    """
    # Extract longitude and latitude
    pred_lon, pred_lat = pred[:, 0], pred[:, 1]
    target_lon, target_lat = target[:, 0], target[:, 1]
    
    # Convert to radians
    pred_lat = torch.deg2rad(pred_lat)
    pred_lon = torch.deg2rad(pred_lon)
    target_lat = torch.deg2rad(target_lat)
    target_lon = torch.deg2rad(target_lon)
    
    # Haversine formula
    dlon = target_lon - pred_lon
    dlat = target_lat - pred_lat
    
    a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371.0  # Radius of Earth in kilometers
    
    # Calculate distance
    distance = r * c
    
    # Return mean distance as loss
    return torch.mean(distance)
