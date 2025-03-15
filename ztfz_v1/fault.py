import numpy as np


class Fault:
    def __init__(self,
                 repaired_time,
                 fault_dict,
                 repair_dict,
                 ):
        self.repaired_time = repaired_time
        self.fault_dict = fault_dict
        self.repair_dict = repair_dict
        self.enabled = True

    @staticmethod
    def bus_grounding_dict(bus, r=0.0, x=0.0):
        """
        获取母线接地故障的修改dict
        :param bus: 故障的母线号(从0开始)
        :param r: 标幺后的接地电阻
        :param x: 标幺后的接地电抗
        :return: dict
        """

        y = 1e20 - 1e20j if r == 0 and x == 0 else 1 / (r + x * 1j)

        return {(bus * 2, bus * 2): y.real,
                (bus * 2 + 1, bus * 2 + 1): y.real,
                (bus * 2, bus * 2 + 1): -y.imag,
                (bus * 2 + 1, bus * 2): y.imag,
                }

    @staticmethod
    def line_cutting_dict(from_bus, to_bus, r, x, b):
        """
        获取断线故障的修改dict
        :param from_bus: 起始母线 (不关心顺序)
        :param to_bus: 终止母线 (不关心顺序)
        :param r: 线路电阻
        :param x: 线路电抗
        :param b: 线路两端导纳
        :return: dict
        """
        y = -1 / (r + 1j * x)
        b2 = b / 2
        return {(from_bus * 2, from_bus * 2): y.real,
                (from_bus * 2, from_bus * 2 + 1): -y.imag + b2,
                (from_bus * 2 + 1, from_bus * 2 + 1): y.real,
                (from_bus * 2 + 1, from_bus * 2): y.imag - b2,

                (to_bus * 2, to_bus * 2): y.real,
                (to_bus * 2, to_bus * 2 + 1): -y.imag + b2,
                (to_bus * 2 + 1, to_bus * 2 + 1): y.real,
                (to_bus * 2 + 1, to_bus * 2): y.imag - b2,

                (from_bus * 2, to_bus * 2): -y.real,
                (from_bus * 2, to_bus * 2 + 1): y.imag,
                (from_bus * 2 + 1, to_bus * 2 + 1): -y.real,
                (from_bus * 2 + 1, to_bus * 2): -y.imag,

                (to_bus * 2, from_bus * 2): -y.real,
                (to_bus * 2, from_bus * 2 + 1): y.imag,
                (to_bus * 2 + 1, from_bus * 2 + 1): -y.real,
                (to_bus * 2 + 1, from_bus * 2): -y.imag,

                }

    def init(self, YY):
        for k, v in self.fault_dict.items():
            YY[k] += v

    def repair(self, t, YY):
        if self.enabled and t >= self.repaired_time:
            # 修复带来的新改变
            for k, v in self.repair_dict.items():
                YY[k] += v
            self.enabled = False

            return True
        else:
            return False


if __name__ == '__main__':
    print(1 / (1e-6 + 1e-6j))