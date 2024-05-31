#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/24 10:15
# @Author : Sunx
# @Last Modified by: Sunx
# @Software: PyCharm

import torch
torch.set_printoptions(profile="full")
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from A_Star_Algorithm import AStar

def LoadMap2Tensor(filename):
    '''
    :param filename: string
    :return: tensor
    '''
    map_data = np.loadtxt(filename, dtype=np.int_, delimiter=' ')
    map_tensor = torch.from_numpy(map_data).unsqueeze(0)
    map_tensor = map_tensor.float()
    return map_tensor

def PatternMatch(X, Kernel, bias, conv):
    '''
    :param X:
    :param Kernel:
    :param bias:
    :param conv:
    :return:
    '''
    device = 'cuda'
    X = X.to(device)
    Kernel = Kernel.to(device)
    bias = bias.to(device)
    conv = conv.to(device)
    with torch.no_grad():
        pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=-1)
        X_ = pad(X)
        X_ = X_.unsqueeze(0)
        Kernel_ = Kernel.unsqueeze(0)
        conv = nn.Conv2d(1, 1, kernel_size=3, bias=True)
        conv.bias.data = bias
        conv.weight.data = Kernel_
        out = conv(X_)
        M = out.gt(0)
    return M

def NonblockPatternMatch(X, Kernel, bias, conv):
    device = 'cuda'
    X = X.to(device)
    Kernel = Kernel.to(device)
    bias = bias.to(device)
    conv = conv.to(device)
    with torch.no_grad():
        pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)
        X_ = pad(X).unsqueeze(0)
        Kernel_ = Kernel.unsqueeze(0)
        conv.bias.data = bias
        conv.weight.data = Kernel_
        out = conv(X_)
        M = out.gt(0)
    return M

def MakeStartTargetMap(MapSize, Start, Target):
    Map = torch.zeros(MapSize)
    Map[0, Start[0], Start[1]] = 1
    Map[0, Target[0], Target[1]] = 1
    return Map

def DeadPatternMatch(X, Kernel, bias, conv):
    dead_kernels = torch.stack([torch.rot90(Kernel, k, [1, 2]) for k in range(4)])  # 包含所有旋转的Kernel
    M_deadends = torch.stack([PatternMatch(X, dk, bias, conv) for dk in dead_kernels])  # 对所有旋转的Kernel进行PatternMatch
    M_deadend = torch.any(M_deadends, dim=0)  # 对所有结果进行逻辑或操作
    return M_deadend

def AvoidablePatternMatch(X, Kernel, bias, conv):
    avoid_kernels = torch.stack([torch.rot90(Kernel, k, [1, 2]) for k in range(4)])  # 包含所有旋转的Kernel
    M_avoids = torch.stack([PatternMatch(X, ak, bias, conv) for ak in avoid_kernels])  # 对所有旋转的Kernel进行PatternMatch
    M_avoid = torch.any(M_avoids, dim=0)  # 对所有结果进行逻辑或操作
    return M_avoid
    # avoid_kernel = Kernel
    # M_avoid_0 = PatternMatch(X, avoid_kernel, bias, conv)
    # avoid_kernel = torch.rot90(avoid_kernel, 1, [1, 2])
    # M_avoid_90 = PatternMatch(X, avoid_kernel, bias, conv)
    # avoid_kernel = torch.rot90(avoid_kernel, 1, [1, 2])
    # M_avoid_180 = PatternMatch(X, avoid_kernel, bias, conv)
    # avoid_kernel = torch.rot90(avoid_kernel, 1, [1, 2])
    # M_avoid_270 = PatternMatch(X, avoid_kernel, bias, conv)
    # M_avoid_1 = torch.logical_or(M_avoid_0, M_avoid_90)
    # M_avoid_2 = torch.logical_or(M_avoid_180, M_avoid_270)
    # M_avoid = torch.logical_or(M_avoid_1, M_avoid_2)
    # return M_avoid

def AlphaPatternMatch(X, Kernel, bias, conv):
    alpha_kernels = torch.stack([torch.rot90(Kernel, k, [1, 2]) for k in range(4)])
    M_alphas = torch.stack([PatternMatch(X, ak, bias, conv) for ak in alpha_kernels])
    M_alpha = torch.any(M_alphas, dim=0)  # 对所有结果进行逻辑或操作
    return M_alpha


def PBGG8N(Map, start, target, maxIter, convKernels, bias):
    Ms = MakeStartTargetMap(Map.shape, start, target).to('cuda')
    i = 0
    M = Map.to('cuda')
    change_num = 5
    conv = nn.Conv2d(1, 1, kernel_size=3, bias=True)
    while (change_num >= 4 or i <= maxIter):
        i += 1
        #如果值是1，则该cell是deadend
        M_deadend = DeadPatternMatch(M, convKernels[0], bias[0], conv)
        # print(M_deadend)

        #如果值是1, 则该cell是avoidable
        M_avoidable1 = AvoidablePatternMatch(M, convKernels[1], bias[1], conv)
        M_avoidable2 = AvoidablePatternMatch(M, convKernels[2], bias[2], conv)
        M_avoidable = torch.logical_or(M_avoidable1, M_avoidable2)
        # print(M_avoidable)
        #
        M_alpha1 = AlphaPatternMatch(M, convKernels[3], bias[3], conv)
        M_alpha2 = AlphaPatternMatch(M, convKernels[4], bias[4], conv)
        M_alpha = torch.logical_or(M_alpha1, M_alpha2)

        M_alpha_S = torch.logical_or(M_alpha, Ms)
        M_alpha_S = torch.squeeze(M_alpha_S, 1)

        M_alpha_S = M_alpha_S.float()
        M_nonblock = NonblockPatternMatch(M_alpha_S, convKernels[5], bias[5], conv)
        M_block = torch.logical_or(M_deadend, M_avoidable)

        M_block = torch.logical_and(M_block, torch.logical_not(torch.logical_or(M_nonblock, Ms)))

        M = M - 2*M_block.int()
        M = torch.squeeze(M, 0)
        M_block = M_block.float()
        change_num = torch.sum(M_block).item()

    return M

def PBGG4N(Map, start, target, maxIter, convKernels, bias):
    device = 'cuda'
    Ms = MakeStartTargetMap(Map.shape, start, target).to(device)
    i = 0
    change_num = 5
    M = Map.to(device)
    conv = nn.Conv2d(1, 1, kernel_size=3, bias=True)
    while (change_num >= 4 or i <= maxIter):
        i += 1
        #如果值是1，则该cell是deadend
        M_deadend = DeadPatternMatch(M, convKernels[0], bias[0], conv)

        #如果值是1, 则该cell是avoidable
        M_avoidable = AvoidablePatternMatch(M, convKernels[1], bias[1], conv)

        #
        M_alpha = AlphaPatternMatch(M, convKernels[2], bias[2], conv)

        M_alpha_S = torch.logical_or(M_alpha, Ms)
        M_alpha_S = torch.squeeze(M_alpha_S, 1)
        M_alpha_S = M_alpha_S.float()

        # M_alpha_S 需要采用0填充
        M_nonblock = NonblockPatternMatch(M_alpha_S, convKernels[3], bias[3], conv)

        M_block = torch.logical_or(M_deadend, M_avoidable)
        M_block = torch.logical_and(M_block, torch.logical_not(torch.logical_or(M_nonblock, Ms)))

        M = M - 2*M_block
        M = torch.squeeze(M, 0)
        M_block = M_block.float()
        change_num = torch.sum(M_block).item()

    return M

def init_8N_kernels():
    kernels_data = [
        [[[-3, -3, -3], [-3, 3, -3], [-1, 0, -1]]],
        [[[1, -4, -4], [4, 4, -4], [1, 1, -4]]],
        [[[1, 1, -4], [4, 4, -4], [1, -4, -4]]],
        [[[1, -4, -4], [1, 4, 4], [1, -4, 4]]],
        [[[1, -4, 4], [1, 4, 4], [1, -4, -4]]],
        [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]
    ]
    kernels = [torch.FloatTensor(data) for data in kernels_data]
    bias_data = [[-17], [-24], [-24], [-22], [-22], [-1]]
    bias = [torch.FloatTensor(data) for data in bias_data]
    return kernels, bias


def init_4N_kernels():
    kernels_data = [
        [[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]],
        [[[0, -1, 0], [-1, 1, 1], [0, 1, 1]]],
        [[[0, 1, -3], [1, 3, 3], [-3, 3, 3]]],
        [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]
    ]
    kernels = [torch.FloatTensor(data) for data in kernels_data]
    bias_data = [[-6], [-5], [-17], [-1]]
    bias = [torch.FloatTensor(data) for data in bias_data]
    return kernels, bias

def drawPicture(img, filename, start, target):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
    fig = plt.figure(figsize=(60, 60))
    np_array = img.detach().cpu().numpy()[0]

    cmap = colors.ListedColormap(['black','white' ])
    # sns.heatmap(map, cmap=cmap, linewidths=0.5, linecolor='black', ax=ax, cbar=False)
    plt.text(x=start[1],
            y=start[0],
            s="S",
            fontdict=dict(fontsize=100, color='r',
                          family='monospace',
                          weight='bold'))

    plt.text(x=target[1],
            y=target[0],
            s="T",
            fontdict=dict(fontsize=100, color='r',
                          family='monospace',
                          weight='bold'))
    plt.imshow(np_array, cmap=cmap, interpolation='nearest')

    plt.xticks()
    plt.yticks()
    plt.grid(True)
    pngname = filename + ".png"
    print(pngname)
    plt.savefig(pngname)
    plt.show()
    plt.close()

def transform_map(map):
    '''
    :param map:
    :return: grid
    '''
    grid[grid == 1] = 0     # 所有为1的部分转化为0
    grid[grid == -1] = 1    # 所有为-1的元素转换为1
    return grid


if __name__ == "__main__":
    filename = "../map_txt/maze512-4-4_copy.txt"
    MapTensor = LoadMap2Tensor(filename)
    torch.cuda.device('cuda')
    start = (488, 448)
    target = (160, 443)
    maxIter = (MapTensor.shape[1] + MapTensor.shape[2]) // 8
    kernel4N, bias4N = init_4N_kernels()
    drawPicture(MapTensor, "maze512-4-4_copy", start, target)
    time1 = time.time()
    M = PBGG4N(MapTensor, start, target, maxIter, kernel4N, bias4N)
    time2 = time.time()
    print("PBGG 4N: ", time2-time1, " s")
    drawPicture(M, "maze512-4-4_copy_after_4N", start, target)

    kernel8N, bias8N = init_8N_kernels()
    time1 = time.time()
    M = PBGG8N(MapTensor, start, target, maxIter, kernel8N, bias8N)
    time2 = time.time()
    print("PBGG 8N:", time2-time1, " s")
    drawPicture(M, "maze512-4-4_copy_after_8N", start, target)

    grid = torch.squeeze(MapTensor).cpu().numpy()
    grid = transform_map(grid)
    print(grid)

    astar = AStar(grid)
    time1 = time.time()
    path = astar.search(start, target)
    time2 = time.time()
    print("astar:", time2 - time1, " ms")
    print("path: ", path)
