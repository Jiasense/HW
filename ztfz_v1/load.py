import numpy as np
from model import Model


class LoadConstant(Model):
    def __init__(self,
                 index,
                 ):
        super().__init__(index)

    def init(self, S0, V0, YY):
        Y0 = S0.conj() / (V0 * V0.conj())
        YY[self.index * 2, self.index * 2] += Y0.real
        YY[self.index * 2 + 1, self.index * 2 + 1] += Y0.real
        YY[self.index * 2, self.index * 2 + 1] += -Y0.imag
        YY[self.index * 2 + 1, self.index * 2] += Y0.imag


class LoadMotor:
    def __init__(self,
                 s0,
                 R1,
                 X1,
                 R2,
                 X2,
                 Rmu,
                 Xmu,
                 Zm,
                 ZM,
                 ):
        self.s0 = s0
        self.R1 = R1
        self.X1 = X1
        self.R2 = R2
        self.X2 = X2
        self.Rmu = Rmu
        self.Xmu = Xmu
        self.Zm = Zm
        self.ZM = ZM

    def step(self, s):
        Z = (self.R1 + 1j * self.X1 +
             (self.Rmu + 1j * self.Xmu) * (self.R2 / s + 1j * self.X2) /
             ((self.Rmu + 1j * self.Xmu) + (self.R2 / s + 1j * self.X2))) * self.Zm / self.ZM
        return Z
