import numpy as np
from utils import xy2dq
from model import Model


class GenSync(Model):
    def __init__(self,
                 index,
                 mode,
                 Ra=0,
                 Xd=0,
                 Xq=0,
                 Xd_=0,
                 Xq_=0,
                 Xd__=0,
                 Xq__=0,
                 Td0_=0,
                 Tq0_=0,
                 Td0__=0,
                 Tq0__=0,
                 TJ=0,
                 D=0,
                 f0=50,
                 ):
        super().__init__(index)
        self.mode = mode

        self.Ra = Ra
        self.Xd = Xd
        self.Xq = Xq
        self.Xq_ = Xq_
        self.Xd_ = Xd_
        self.Xd__ = Xd__
        self.Xq__ = Xq__

        self.Td0_ = Td0_
        self.Tq0_ = Tq0_
        self.Td0__ = Td0__
        self.Tq0__ = Tq0__
        self.TJ = TJ
        self.D = D

        self.Efd = 0

        self.Eq = 0
        self.Ed = 0
        self.Eq_ = 0
        self.Ed_ = 0
        self.Eq__ = 0
        self.Ed__ = 0

        self.Vx = 0
        self.Vy = 0
        self.Ix = 0
        self.Iy = 0
        self.Ixx = 0  # 虚拟节点注入电流
        self.Iyy = 0  # 虚拟节点注入电流
        self.Id = 0
        self.Iq = 0
        self.delta = 0

        self.Gx = 0
        self.Gy = 0
        self.Bx = 0
        self.By = 0

        self.gx = 0
        self.gy = 0
        self.bx = 0
        self.by = 0

        self.Pm = 0
        self.Pe = 0

        self.f0 = f0
        self.w = np.ones(len(index), dtype=np.float64)
        self.ws = (2 * np.pi * f0 * np.ones(len(index))).astype(np.float64)

    @staticmethod
    def implicit_IE(step):
        from sympy import symbols, solve, lambdify
        from sympy import Eq as Equation
        from utils import Params

        dEq_dt = lambda x: (x.Efd - (x.Eq_ + (x.Xd - x.Xd_) * x.Id)) / x.Td0_
        dEd_dt = lambda x: (-x.Ed_ + (x.Xq - x.Xq_) * x.Iq) / x.Tq0_
        dEq__dt = lambda x: \
            (- x.Eq__ - (x.Xd_ - x.Xd__) * x.Id + x.Eq_ + x.Td0__ * x.dEq_dt) / x.Td0__
        dEd__dt = lambda x: \
            (- x.Ed__ + (x.Xq_ - x.Xq__) * x.Iq + x.Ed_ + x.Tq0__ * x.dEd_dt) / x.Tq0__
        ddeltadt = lambda x: x.ws * (x.w - 1)
        dwdt = lambda x: (x.Pm - x.Pe) / x.TJ

        con = ['Pm', 'Efd', 'TJ', 'ws',
               'Xd', 'Xd_', 'Xd__',
               'Xq', 'Xq_', 'Xq__',
               'Td0_', 'Td0__', 'Tq0_', 'Tq0__']
        dym = ['Id', 'Iq', 'Pe',
               'Ed_', 'Ed__',
               'Eq_', 'Eq__',
               'w', 'delta',
               'dEd_dt', 'dEq_dt',
               ]

        param_t = Params(**{n: symbols(n) for n in con},
                         **{n: symbols('t_' + n) for n in dym},
                         )
        param_t1 = Params(**{n: param_t[n] for n in con},
                          **{n: symbols('t1_' + n) for n in dym},
                          )

        ld = locals()
        eqs = {name: Equation(param_t1[name],
                              param_t[name] +
                              (ld[f'd{name}dt'](param_t) +
                               ld[f'd{name}dt'](param_t1)) / 2 * step)
               for name in ['Ed_', 'Ed__', 'Eq_', 'Eq__', 'w', 'delta']}

        funcs = {name: lambdify((*param_t.values(), *[param_t1[n] for n in dym]),
                                solve(eqs[name], param_t1[name])[0],
                                np)
                 for name in eqs}

        # fs = {name: lambda pt, pt1: funcs[name](*[pt[key] for key in con],
        #                                         *[pt[key] for key in dym],
        #                                         *[pt1[key] for key in dym],
        #                                         )
        #       for name in funcs}

        return dict(Ed_=lambda pt, pt1: funcs['Ed_'](*[pt[key] for key in con],
                                                     *[pt[key] for key in dym],
                                                     *[pt1[key] for key in dym],
                                                     ),
                    Ed__=lambda pt, pt1: funcs['Ed__'](*[pt[key] for key in con],
                                                       *[pt[key] for key in dym],
                                                       *[pt1[key] for key in dym],
                                                       ),
                    Eq_=lambda pt, pt1: funcs['Eq_'](*[pt[key] for key in con],
                                                     *[pt[key] for key in dym],
                                                     *[pt1[key] for key in dym],
                                                     ),
                    Eq__=lambda pt, pt1: funcs['Eq__'](*[pt[key] for key in con],
                                                       *[pt[key] for key in dym],
                                                       *[pt1[key] for key in dym],
                                                       ),
                    w=lambda pt, pt1: funcs['w'](*[pt[key] for key in con],
                                                 *[pt[key] for key in dym],
                                                 *[pt1[key] for key in dym],
                                                 ),
                    delta=lambda pt, pt1: funcs['delta'](*[pt[key] for key in con],
                                                         *[pt[key] for key in dym],
                                                         *[pt1[key] for key in dym],
                                                         ),
                    )

    def init(self, S0, V0, YY):
        # if self.mode == 0:
        #     Xd = self.Xd__
        #     Xq = self.Xq__
        # elif self.mode == 1:
        #     Xd = self.Xd_
        #     Xq = self.Xq_
        # elif self.mode == 2:
        #     Xd = self.Xd_
        #     Xq = self.Xq
        # else:
        #     Xd = self.Xd_
        #     Xq = self.Xd_
        Xd = self.Xd
        Xq = self.Xq

        I0 = S0.conj() / V0.conj()

        EQ0 = V0 + (self.Ra + 1j * Xq) * I0
        self.delta = np.arctan(EQ0.imag / EQ0.real)

        Vd0, Vq0 = xy2dq(V0.real, V0.imag, self.delta)
        Id0, Iq0 = xy2dq(I0.real, I0.imag, self.delta)
        self.Eq_ = Vq0 + self.Ra * Iq0 + self.Xd_ * Id0
        self.Ed_ = Vd0 + self.Ra * Id0 - self.Xq_ * Iq0
        self.Eq__ = Vq0 + self.Ra * Iq0 + self.Xd__ * Id0
        self.Ed__ = Vd0 + self.Ra * Id0 - self.Xq__ * Iq0

        self.Pe = S0.real + (I0.real ** 2 + I0.imag ** 2) * self.Ra
        self.Pm = self.Pe + self.D

        self.Efd = Vq0 + self.Ra * Iq0 + Xd * Id0

        self.Vx = V0.real
        self.Vy = V0.imag
        self.Ix = I0.real
        self.Iy = I0.imag

        self.Id, self.Iq = xy2dq(self.Ix, self.Iy, self.delta)

    def update_YY(self, YY):
        # 计算虚拟电缆并更新导纳矩阵
        if self.mode == 0:
            self.Ed = self.Ed__
            self.Eq = self.Eq__
            Xd = self.Xd__
            Xq = self.Xq__
        elif self.mode == 1:
            self.Ed = self.Ed_
            self.Eq = self.Eq_
            Xd = self.Xd_
            Xq = self.Xq_
        elif self.mode == 2:
            self.Ed = 0
            self.Eq = self.Eq_
            Xd = self.Xd_
            Xq = self.Xq
        else:
            self.Ed = 0
            self.Eq = self.Eq_
            Xd = self.Xd_
            Xq = self.Xd_

        # 计算新阻抗
        self.Gx = (self.Ra - (Xd - Xq) * np.sin(self.delta) * np.cos(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.Bx = (Xd * np.cos(self.delta) ** 2 + Xq * np.sin(self.delta) ** 2) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.Gy = (self.Ra + (Xd - Xq) * np.sin(self.delta) * np.cos(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.By = (-Xd * np.sin(self.delta) ** 2 - Xq * np.cos(self.delta) ** 2) / \
                  (self.Ra ** 2 + Xd * Xq)

        self.gx = (self.Ra * np.sin(self.delta) - Xd * np.cos(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.gy = (self.Ra * np.sin(self.delta) - Xq * np.cos(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.bx = (self.Ra * np.cos(self.delta) + Xq * np.sin(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)
        self.by = (-self.Ra * np.cos(self.delta) - Xd * np.sin(self.delta)) / \
                  (self.Ra ** 2 + Xd * Xq)

        self.Ixx = self.gx * self.Ed + self.bx * self.Eq
        self.Iyy = self.by * self.Ed + self.gy * self.Eq

        # 更新分块Y矩阵
        YY[self.index * 2, self.index * 2] += self.Gx
        YY[self.index * 2 + 1, self.index * 2 + 1] += self.Gy
        YY[self.index * 2, self.index * 2 + 1] += self.Bx
        YY[self.index * 2 + 1, self.index * 2] += self.By

    def calculate_V(self, YY):
        I = np.zeros(YY.shape[0], dtype=np.float64)
        I[self.index * 2] = self.Ixx
        I[self.index * 2 + 1] = self.Iyy
        return np.linalg.inv(YY) @ I

    def set_V(self, V):
        self.Vx = V[self.index * 2]
        self.Vy = V[self.index * 2 + 1]

        # 更新定子电流
        self.Ix = self.gx * self.Ed + self.bx * self.Eq - (self.Gx * self.Vx + self.Bx * self.Vy)
        self.Iy = self.by * self.Ed + self.gy * self.Eq - (self.By * self.Vx + self.Gy * self.Vy)
        self.Id, self.Iq = xy2dq(self.Ix, self.Iy, self.delta)

        # 简化公式
        self.Pe = (self.Vx * self.Ix + self.Vy * self.Iy) - (self.Ix ** 2 + self.Iy ** 2) * self.Ra

        # 非简化公式, ?
        # Vd, Vq = xy2dq(self.Vx, self.Vy, self.delta)
        # fai_q = - self.Ra * self.Iq - Vd
        # fai_d = self.Ra * self.Id + Vq
        # self.Pe = fai_d * self.Iq - fai_q * self.Id

    def set(self,
            Ed_=0,
            Ed__=0,
            Eq_=0,
            Eq__=0,
            w=0,
            delta=0,
            ):
        self.Ed_ = Ed_
        self.Eq_ = Eq_
        self.Ed__ = Ed__
        self.Eq__ = Eq__
        self.w = w
        self.delta = delta

    def step(self,
             step_t,
             dEd_dt=0,
             dEd__dt=0,
             dEq_dt=0,
             dEq__dt=0,
             dwdt=0,
             ddeltadt=0,
             ):
        if self.mode <= 1:
            if dEq_dt is 0:
                dEq_dt = (self.Efd - (self.Eq_ + (self.Xd - self.Xd_) * self.Id)) / self.Td0_
            if dEd_dt is 0:
                dEd_dt = (-self.Ed_ + (self.Xq - self.Xq_) * self.Iq) / self.Tq0_

        if self.mode == 0:
            if dEq__dt is 0:
                dEq__dt = (- self.Eq__ - (self.Xd_ - self.Xd__) * self.Id + self.Eq_ + self.Td0__ * dEq_dt) / self.Td0__
            if dEd__dt is 0:
                dEd__dt = (- self.Ed__ + (self.Xq_ - self.Xq__) * self.Iq + self.Ed_ + self.Tq0__ * dEd_dt) / self.Tq0__

        if ddeltadt is 0:
            ddeltadt = self.ws * (self.w - 1)
        if dwdt is 0:
            dwdt = (self.Pm - self.Pe) / self.TJ

        if self.mode <= 1:
            self.Eq_ = self.Eq_ + dEq_dt * step_t
            self.Ed_ = self.Ed_ + dEd_dt * step_t
        if self.mode == 0:
            self.Eq__ = self.Eq__ + dEq__dt * step_t
            self.Ed__ = self.Ed__ + dEd__dt * step_t
        self.delta = self.delta + ddeltadt * step_t
        self.w = self.w + dwdt * step_t

        return (dEd_dt,
                dEd__dt,
                dEq_dt,
                dEq__dt,
                dwdt,
                ddeltadt,
                )

    def check_result(self, dd, YY):
        l = dd['Ed__'].shape[0]

        if self.mode == 0:
            Ed = dd['Ed__']
            Eq = dd['Eq__']
            Xd = self.Xd__
            Xq = self.Xq__
        elif self.mode == 1:
            Ed = dd['Ed_']
            Eq = dd['Eq_']
            Xd = self.Xd_
            Xq = self.Xq_
        elif self.mode == 2:
            Ed = 0
            Eq = dd['Eq_']
            Xd = self.Xd_
            Xq = self.Xq
        else:
            Ed = 0
            Eq = dd['Eq_']
            Xd = self.Xd_
            Xq = self.Xd_

        from inspect import ismethod
        for key, value in self.__dict__.items():
            if not ismethod(value):
                print(key, value)

        # 计算新阻抗
        Gx = (self.Ra - (Xd - Xq) * np.sin(dd['delta']) * np.cos(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)
        Bx = (Xd * np.cos(dd['delta']) ** 2 + Xq * np.sin(dd['delta']) ** 2) / \
             (self.Ra ** 2 + Xd * Xq)
        Gy = (self.Ra + (Xd - Xq) * np.sin(dd['delta']) * np.cos(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)
        By = (-Xd * np.sin(dd['delta']) ** 2 - Xq * np.cos(dd['delta']) ** 2) / \
             (self.Ra ** 2 + Xd * Xq)

        gx = (self.Ra * np.sin(dd['delta']) - Xd * np.cos(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)
        gy = (self.Ra * np.sin(dd['delta']) - Xq * np.cos(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)
        bx = (self.Ra * np.cos(dd['delta']) + Xq * np.sin(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)
        by = (-self.Ra * np.cos(dd['delta']) - Xd * np.sin(dd['delta'])) / \
             (self.Ra ** 2 + Xd * Xq)

        Ixx = gx * Ed + bx * Eq
        Iyy = by * Ed + gy * Eq

        YY = np.stack([YY] * l, axis=0)
        YY[:, self.index * 2, self.index * 2] += Gx
        YY[:, self.index * 2 + 1, self.index * 2 + 1] += Gy
        YY[:, self.index * 2, self.index * 2 + 1] += Bx
        YY[:, self.index * 2 + 1, self.index * 2] += By

        I = np.zeros((l, YY.shape[-1], 1), dtype=np.float64)
        I[:, self.index * 2, 0] = Ixx
        I[:, self.index * 2 + 1, 0] = Iyy
        V = np.linalg.inv(YY) @ I
        Vx = V[:, self.index * 2, 0]
        Vy = V[:, self.index * 2 + 1, 0]

        Ix = gx * Ed + bx * Eq - (Gx * Vx + Bx * Vy)
        Iy = by * Ed + gy * Eq - (By * Vx + Gy * Vy)
        Id, Iq = xy2dq(Ix, Iy, dd['delta'])
        Pe = (Vx * Ix + Vy * Iy) - (Ix ** 2 + Iy ** 2) * self.Ra

        for name in ['Ix', 'Iy', 'Vx', 'Vy', 'Pe', 'Id', 'Iq']:
            print(name, abs(locals()[name] - dd[name]).mean())

        Vd, Vq = xy2dq(Vx, Vy, dd['delta'])
        fai_q = 1 / self.w * (- self.Ra * Iq - Vd)
        fai_d = 1 / self.w * (self.Ra * Id + Vq)
        Pe = fai_d * Iq - fai_q * Id


if __name__ == '__main__':
    GenSync.implicit_IE(0.01)
