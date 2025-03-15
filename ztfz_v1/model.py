import numpy as np
from copy import deepcopy


class Model:
    # 基础模型
    def __init__(self,
                 index,
                 ):
        self.index = np.array(index, dtype=int)  # bus的index

    def init(self, S0, V0, YY):
        """
        初始化自身与导纳矩阵
        :param S0: self.index对应的节点的S, dtype=np.complex128
        :param V0: self.index对应的节点的V, dtype=np.complex128
        :param YY: 导纳矩阵, dtype=np.float64
        :return: None
        """
        return

    def update_YY(self, YY):
        """
        根据自身状态修改导纳矩阵
        :param YY: 导纳矩阵
        :return: None
        """
        return

    def set_V(self, V):
        """
        从所有节点的电压中设置自身节点的电压
        :param V: 所有节点的电压array
        :return: None
        """
        return

    def copy(self):
        """
        复制自身
        :return: 复制后的自身
        """
        return deepcopy(self)

    def step(self, step_t, *args, **kwargs):
        """
        算自身有的梯度, 并按照时间间隔更新自身
        :param step_t: 时间间隔
        :param args: 指定自身的梯度值
        :param kwargs: 指定自身的梯度值
        :return:
        """
        return (0.0,)
