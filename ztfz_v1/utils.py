import numpy as np
from pypower.api import ext2int, loadcase, makeYbus


def xy2dq(x, y, delta):
    # xy坐标系转dq坐标系, VI通用
    d = x * np.sin(delta) - y * np.cos(delta)
    q = x * np.cos(delta) + y * np.sin(delta)
    return d, q


def check_dq(x, y, d, q, delta):
    _d, _q = xy2dq(x, y, delta)

    print(np.mean(np.abs(d - _d)), np.mean(np.abs(q - _q)))

#对电网数据进行归一化处理，将节点编号映射到连续的索引
#grid: 电网数据字典，包含节点、支路和发电机信息
#bus_mapping: 原始节点编号到新索引的映射字典

def normalize_grid(grid):
    bus_mapping = {bus: i for i, bus in enumerate(grid['bus'][:, 0])}
    grid['bus'][:, 0] = np.arange(grid['bus'].shape[0])
    grid['gen'][:, 0] = np.array([bus_mapping[bus] for bus in grid['gen'][:, 0]])
    grid['branch'][:, 0] = np.array([bus_mapping[bus] for bus in grid['branch'][:, 0]])
    grid['branch'][:, 1] = np.array([bus_mapping[bus] for bus in grid['branch'][:, 1]])
    return grid, bus_mapping

#生成IEEE 9节点系统的电网数据
def ieee9():
    bus = np.zeros((9, 12))
    bus[:, 0] = np.arange(1, 10)  #节点编号
    bus[:, 1] = 1
    bus[0, 1] = 3
    bus[1, 1] = 2
    bus[2, 1] = 2
    bus[:, [6, 7]] = 1
    bus[:, 10] = 1
    # bus[:, 11] = 1.1
    # bus[:, 12] = 0.9
    bus[4, [2, 3]] = np.array([1.25, 0.5]) * 100
    bus[5, [2, 3]] = np.array([0.9, 0.3]) * 100
    bus[7, [2, 3]] = np.array([1.0, 0.35]) * 100
    bus[:, 9] = 230
    bus[1, 9] = 16.5
    bus[2, 9] = 18
    bus[3, 9] = 13.8

    branch = np.zeros((9, 13)) 
    branch[:, 11] = -360
    branch[:, 12] = 360
    branch[:, 10] = 1
    branch[0, :5] = [4, 5, 0.01, 0.085, 0.088 * 2]
    branch[1, :5] = [4, 6, 0.017, 0.092, 0.079 * 2]
    branch[2, :5] = [5, 7, 0.032, 0.161, 0.153 * 2]
    branch[3, :5] = [6, 9, 0.039, 0.170, 0.179 * 2]
    branch[4, :5] = [7, 8, 0.0085, 0.072, 0.0745 * 2]
    branch[5, :5] = [8, 9, 0.0119, 0.1008, 0.1045 * 2]
    branch[6, :5] = [1, 4, 0.0, 0.0576, 0.0]
    branch[7, :5] = [2, 7, 0.0, 0.0625, 0.0]
    branch[8, :5] = [3, 9, 0.0, 0.0586, 0.0]
    branch[6:, 8] = 1

    gen = np.zeros((3, 21))
    gen[:, 0] = 1, 2, 3
    gen[:, 1] = np.array([0.7164, 1.63, 0.85]) * 100
    gen[:, 2] = np.array([0.2705, 0.0665, -0.1086]) * 100
    gen[:, 5] = 1.04, 1.025, 1.025
    gen[:, 6] = 100
    gen[:, 7] = 1

    grid = dict(bus=bus,
                branch=branch,
                gen=gen,
                baseMVA=100,
                )

    # r, s = runpf(grid,
    #              ppopt=ppoption(VERBOSE=0,
    #                             OUT_ALL=0,
    #                             ))
    # print(r['bus'][:, [7, 8]])
    # print(r['gen'][:, [1, 2]])
    return grid



def get_y(grid):
    ppc = loadcase(grid)
    ppc = ext2int(ppc)
    return makeYbus(ppc['baseMVA'],
                    ppc['bus'],
                    ppc['branch'])[0].toarray()


class Params(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__