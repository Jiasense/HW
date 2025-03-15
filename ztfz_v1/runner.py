import numpy as np
from pypower.api import runpf, ppoption
from gen import GenSync
from utils import get_y
from load import LoadConstant
from fault import Fault


class Runner:
    def __init__(self,
                 grid,
                 step_t,  # 时间步长
                 max_t,
                 gen_model: GenSync,
                 load_model: LoadConstant,
                 fault_model: Fault,
                 method='IE',
                 gen_save_names=('delta',),
                 bus_save_names=(),
                 ):
        self.step_t = step_t
        self.max_t = max_t
        self.now_t = 0
        self.precision = self.get_decimal_precision(self.step_t)

        grid, success = runpf(grid,
                              ppopt=ppoption(VERBOSE=0,
                                             OUT_ALL=0,
                                             ))
        assert success, '原模型无法计算潮流！！'

        self.baseMVA = grid['baseMVA']
        self.bus = grid['bus'].astype(np.float64)
        self.gen = grid['gen'].astype(np.float64)
        self.branch = grid['branch'].astype(np.float64)
        self.Y = get_y(grid).astype(np.complex128)

        self.bus_num = self.bus.shape[0]
        self.gen_num = self.gen.shape[0]
        self.branch_num = self.branch.shape[0]

        self.gen_indices = (self.gen[:, 0] - 1).astype(int)
        self.load_indices = np.where((self.bus[:, 2] != 0) | (self.bus[:, 3] != 0))

        self.gen_model = gen_model
        self.load_model = load_model
        self.fault_model = fault_model

        self.YY = None
        self.YY_fault = None
        self.YY_repair = None

        self.YY_initial = None

        self.method = method
        assert getattr(self, f'step_{method}')

        self.gen_save_names = gen_save_names
        self.bus_save_names = bus_save_names
        self.gen_buffer = {name: np.zeros((0, self.gen_num))
                           for name in gen_save_names}
        self.bus_buffer = {name: np.zeros((0, self.bus_num))
                           for name in bus_save_names}

        self.iie_funcs = self.gen_model.implicit_IE(self.step_t)

    def _get_gen_v(self):
        v = self.bus[self.gen_indices, 7]
        a = self.bus[self.gen_indices, 8] * np.pi / 180
        return v * np.cos(a) + 1j * v * np.sin(a)

    def _get_gen_s(self):
        return (self.gen[:, 1] + 1j * self.gen[:, 2]) / 100

    def _get_load_v(self):
        v = self.bus[self.load_indices, 7]
        a = self.bus[self.load_indices, 8] * np.pi / 180
        return v * np.cos(a) + 1j * v * np.sin(a)

    def _get_load_s(self):
        return (self.bus[self.load_indices, 2] + 1j * self.bus[self.load_indices, 3]) / 100

    def run(self):
        Y = self.Y.copy()
        YY = np.zeros((self.bus_num * 2, self.bus_num * 2))
        YY[slice(0, self.bus_num * 2, 2), slice(0, self.bus_num * 2, 2)] = Y.real
        YY[slice(1, self.bus_num * 2, 2), slice(1, self.bus_num * 2, 2)] = Y.real
        YY[slice(0, self.bus_num * 2, 2), slice(1, self.bus_num * 2, 2)] = -Y.imag
        YY[slice(1, self.bus_num * 2, 2), slice(0, self.bus_num * 2, 2)] = Y.imag

        self.YY = YY

        self.gen_model.init(self._get_gen_s(), self._get_gen_v(), self.YY)
        self.load_model.init(self._get_load_s(), self._get_load_v(), self.YY)

        self.YY_initial = self.YY.copy()
        self.fault_model.init(self.YY)

        self.YY_fault = self.YY.copy()

        # self._record_bus()
        # self._record_gen()
        while self.now_t <= self.max_t:
            if self.fault_model.repair(self.now_t, self.YY_initial):
                self.YY = self.YY_initial.copy()
                self.YY_repair = self.YY.copy()
            getattr(self, f'step_{self.method}')()
            self.now_t = round(self.now_t + self.step_t, self.precision)

        return self.gen_buffer

    def _record_bus(self, V=None):
        # 记录母线/节点数据
        if V is None:
            if 'V' in self.bus_save_names:
                self.bus_buffer['V'] = self.bus[:, 7].reshape(1, -1)
            if 'a' in self.bus_save_names:
                self.bus_buffer['a'] = (self.bus[:, 8] * np.pi / 180).reshape(1, -1)
            if 'Vx' in self.bus_save_names:
                self.bus_buffer['Vx'] = (self.bus[:, 7] * np.cos(self.bus[:, 8] * np.pi / 180)).reshape(1, -1)
            if 'Vy' in self.bus_save_names:
                self.bus_buffer['Vy'] = (self.bus[:, 7] * np.sin(self.bus[:, 8] * np.pi / 180)).reshape(1, -1)
            return

        Vx = V[::2]
        Vy = V[1:len(V):2]
        if 'V' in self.bus_save_names:
            self.bus_buffer['V'] = np.vstack([self.bus_buffer['V'],
                                              (Vx ** 2 + Vy ** 2) ** 0.5])
        if 'a' in self.bus_save_names:
            self.bus_buffer['a'] = np.vstack([self.bus_buffer['a'],
                                              np.arctan(Vy / Vx)])
        if 'Vx' in self.bus_save_names:
            self.bus_buffer['Vx'] = np.vstack([self.bus_buffer['Vx'],
                                               Vx])
        if 'Vy' in self.bus_save_names:
            self.bus_buffer['Vy'] = np.vstack([self.bus_buffer['Vy'],
                                               Vy])

    def _record_gen(self):
        # 记录发电机数据
        for name in self.gen_save_names:
            self.gen_buffer[name] = np.vstack([self.gen_buffer[name],
                                               getattr(self.gen_model, name, 0)])

    def step_IE(self):
        YY = self.YY.copy()

        self.gen_model.update_YY(YY)
        self.load_model.update_YY(YY)

        V = self.gen_model.calculate_V(YY)
        self._record_bus(V)

        self.gen_model.set_V(V)
        self.load_model.set_V(V)

        self._record_gen()
        ################################################################################################################
        gen_model = self.gen_model.copy()
        load_model = self.load_model.copy()

        grads0 = []
        grads0.extend(gen_model.step(self.step_t))
        grads0.extend(load_model.step(self.step_t))
        ################################################################################################################
        YY = self.YY.copy()
        gen_model.update_YY(YY)
        load_model.update_YY(YY)

        V = gen_model.calculate_V(YY)
        gen_model.set_V(V)
        load_model.set_V(V)

        grads1 = []
        grads1.extend(gen_model.step(self.step_t))
        grads1.extend(load_model.step(self.step_t))
        ################################################################################################################

        self.gen_model.step(self.step_t,
                            *[(grads0[i] + grads1[i]) / 2
                              for i in range(len(grads0) - 1)]
                            )
        self.load_model.step(self.step_t,
                             (grads0[-1] + grads1[-1]) / 2)

    def step_IIE(self):
        from utils import Params
        YY = self.YY.copy()

        self.gen_model.update_YY(YY)
        self.load_model.update_YY(YY)

        V = self.gen_model.calculate_V(YY)
        self._record_bus(V)

        self.gen_model.set_V(V)
        self.load_model.set_V(V)

        self._record_gen()
        ################################################################################################################
        gen_model = self.gen_model.copy()
        load_model = self.load_model.copy()

        param_gen0 = Params(**{key: getattr(gen_model, key) for key in dir(gen_model) if key[0] != '_'})

        grads0 = []
        grads0.extend(gen_model.step(self.step_t))
        grads0.extend(load_model.step(self.step_t))

        for i, k in enumerate(self.iie_funcs):
            param_gen0[f'd{k}dt'] = grads0[i]
        ################################################################################################################
        YY = self.YY.copy()
        gen_model.update_YY(YY)
        load_model.update_YY(YY)

        V = gen_model.calculate_V(YY)
        gen_model.set_V(V)
        load_model.set_V(V)

        param_gen1 = Params(**{key: getattr(gen_model, key) for key in dir(gen_model) if key[0] != '_'})

        grads1 = []
        grads1.extend(gen_model.step(self.step_t))
        grads1.extend(load_model.step(self.step_t))

        for i, k in enumerate(self.iie_funcs):
            param_gen1[f'd{k}dt'] = grads1[i]
        ################################################################################################################
        self.gen_model.set(*[self.iie_funcs[key](param_gen0, param_gen1)
                             for key in self.iie_funcs])

        self.load_model.step(self.step_t,
                             (grads0[-1] + grads1[-1]) / 2)

    def step_RK4(self):
        YY = self.YY.copy()

        self.gen_model.update_YY(YY)
        self.load_model.update_YY(YY)

        V = self.gen_model.calculate_V(YY)
        self._record_bus(V)

        self.gen_model.set_V(V)
        self.load_model.set_V(V)

        self._record_gen()
        ################################################################################################################
        gen_model = self.gen_model.copy()
        load_model = self.load_model.copy()

        grads0 = []
        grads0.extend(gen_model.step(self.step_t))
        grads0.extend(load_model.step(self.step_t))
        ################################################################################################################
        YY = self.YY.copy()
        gen_model.update_YY(YY)
        load_model.update_YY(YY)

        V = gen_model.calculate_V(YY)
        gen_model.set_V(V)
        load_model.set_V(V)

        grads1 = []
        grads1.extend(gen_model.step(self.step_t))
        grads1.extend(load_model.step(self.step_t))
        ################################################################################################################
        YY = self.YY.copy()
        gen_model.update_YY(YY)
        load_model.update_YY(YY)

        V = gen_model.calculate_V(YY)
        gen_model.set_V(V)
        load_model.set_V(V)

        grads2 = []
        grads2.extend(gen_model.step(self.step_t))
        grads2.extend(load_model.step(self.step_t))
        ################################################################################################################
        YY = self.YY.copy()
        gen_model.update_YY(YY)
        load_model.update_YY(YY)

        V = gen_model.calculate_V(YY)
        gen_model.set_V(V)
        load_model.set_V(V)

        grads3 = []
        grads3.extend(gen_model.step(self.step_t))
        grads3.extend(load_model.step(self.step_t))
        ################################################################################################################
        self.gen_model.step(self.step_t,
                            *[(grads0[i] + 2 * grads1[i] + 2 * grads2[i] + grads3[i]) / 6
                              for i in range(len(grads0) - 1)]
                            )
        self.load_model.step(self.step_t,
                             (grads0[-1] + 2 * grads1[-1] + 2 * grads2[-1] + grads3[-1]) / 6)

    def get_result_df(self):
        import pandas as pd

        gen_arr = np.hstack([each for each in self.gen_buffer.values()])

        if self.bus_save_names:
            bus_arr = np.hstack([each for each in self.bus_buffer.values()])
            arr = np.hstack([gen_arr, bus_arr])
        else:
            arr = gen_arr

        df = pd.DataFrame(arr,
                          index=np.round(np.arange(arr.shape[0]) * self.step_t,
                                         self.precision),
                          columns=[f'{name} gen{i}' for name in self.gen_save_names for i in range(self.gen_num)] +
                                  [f'{name} bus{i}' for name in self.bus_save_names for i in range(self.bus_num)],
                          )
        df.index.name = 'time'

        return df

    @staticmethod
    def get_decimal_precision(number):
        # 取绝对值以处理负数
        abs_number = abs(number)
        # 计算对数
        log_value = np.log10(abs_number)
        # 如果对数是负数，则取其绝对值并向上取整
        if log_value < 0:
            precision = int(np.ceil(-log_value))
        else:
            precision = 0
        return precision


if __name__ == '__main__':
    from sympy import symbols, Eq, solve, lambdify

    Eq_t, Eq_t1, Idt, Idt1 = symbols('Eq_t Eq_t1 Idt Idt1')
    func = lambda x, y: 1 - (x + 0.2 * y)
    eq = Eq(Eq_t1, Eq_t + 1 / 2 * 0.01 * (func(Eq_t, Idt) + func(Eq_t1, Idt1)))
    eq_solve = solve(eq, Eq_t1)[0]
    f = lambdify((Eq_t, Idt, Idt1), eq_solve, 'numpy')
    print(f(1, 2, 3))
