import numpy as np
import pandas as pd
from gen import GenSync
from load import LoadConstant
from fault import Fault
from drawing import Drawing
from runner import Runner
from utils import ieee9


def test():
    grid = ieee9()
    gen_model = GenSync((grid['gen'][:, 0] - 1).astype(int),
                        mode=0,
                        Ra=np.array([0, 0, 0]),
                        Xd=np.array([0.146, 0.8958, 1.3125]),
                        Xd_=np.array([0.0608, 0.1198, 0.1813]),
                        Xd__=np.array([0.04, 0.089, 0.107]),
                        Xq=np.array([0.0969, 0.8645, 1.2578]),
                        Xq_=np.array([0.1, 0.1969, 0.25]),
                        Xq__=np.array([0.06, 0.089, 0.107]),
                        Td0_=np.array([8.96, 6, 5.89]),
                        Td0__=np.array([6, 6, 6]),
                        Tq0_=np.array([0.54, 0.42, 0.6]),
                        Tq0__=np.array([0.535, 0.535, 0.535]),
                        TJ=np.array([47.28, 12.80, 6.02]),
                        f0=50,
                        )
    load_model = LoadConstant(np.where((grid['bus'][:, 2] != 0) | (grid['bus'][:, 3] != 0)),
                              )
    fault = Fault(0.3,
                  Fault.bus_grounding_dict(6),
                  {},
                  )
    names = ['Eq_', 'Eq__', 'Ed_', 'Ed__', 'w', 'delta', 'Pe', 'Id', 'Iq',
             'Ix', 'Iy', 'Vx', 'Vy', 'Efd', 'Pm']
    runner = Runner(grid,
                    step_t=1e-3,
                    max_t=5,
                    gen_model=gen_model,
                    load_model=load_model,
                    fault_model=fault,
                    gen_save_names=names,
                    bus_save_names=['V', 'a'],
                    method='IE',
                    )
    runner.run()
    # runner.gen_model._check_result({key: value[-100:]
    #                                 for key, value in runner.gen_buffer.items()},
    #                                runner.YY_initial,
    #                                )

    df = runner.get_result_df()
    compare(df, read_excel())


def compare(df, true_df):
    print(df.shape, true_df.shape)
    for col in sorted(true_df.columns):
        # if 'Pe' in col:
        a = true_df.loc[:, col].to_numpy()
        b = df.loc[:, col].to_numpy()
        print(col, abs(a - b).mean(), abs(a - b).max())
        Drawing(-4).draw(dict(y=a,
                              label='true',
                              color='orange',
                              ),
                         dict(y=b,
                              label='predict',
                              color='blue',
                              ),
                         title=col,
                         )

    true_dd1 = (true_df.loc[:, 'delta gen1'] - true_df.loc[:, 'delta gen0']).to_numpy()[1:]
    true_dd2 = (true_df.loc[:, 'delta gen2'] - true_df.loc[:, 'delta gen0']).to_numpy()[1:]

    dd1 = (df.loc[:, 'delta gen1'] - df.loc[:, 'delta gen0']).to_numpy()[1:]
    dd2 = (df.loc[:, 'delta gen2'] - df.loc[:, 'delta gen0']).to_numpy()[1:]

    print(abs(true_dd1 - dd1).mean(), abs(true_dd1 - dd1).max())
    print(abs(true_dd2 - dd2).mean(), abs(true_dd2 - dd2).max())

    Drawing(-4).draw(dict(y=true_dd1,
                          label='true',
                          color='orange',
                          ),
                     dict(y=dd1,
                          label='predict',
                          color='blue',
                          ),
                     title='delta1 - delta0',
                     )

    Drawing(-4).draw(dict(y=true_dd2,
                          label='true',
                          color='orange',
                          ),
                     dict(y=dd2,
                          label='predict',
                          color='blue',
                          ),
                     title='delta2 - delta0',
                     )


def read_excel():
    # 读PSASP输出文件

    # 故障切除时间为0.3s, 失稳
    df: pd.DataFrame = pd.read_excel('curveData20250116144016.xls')

    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    def __col_map(col):
        if '发电1' in col:
            gen = 0
        elif '发电2' in col:
            gen = 1
        elif '发电3' in col:
            gen = 2
        else:
            raise ValueError(col)

        obj = col.split('(')[0]
        if obj == "E'd":
            obj = 'Ed_'
        elif obj == "E'q":
            obj = 'Eq_'
        elif obj == 'E"d':
            obj = 'Ed__'
        elif obj == 'E"q':
            obj = 'Eq__'
        elif obj == 'δ':
            obj = 'delta'
        elif obj == 'ω':
            obj = 'w'
        elif obj in ['Efd', 'Pm']:
            pass
        elif obj == 'P':
            obj = 'Pe'
        else:
            raise ValueError(obj)

        return f'{obj} gen{gen}'

    df.columns = [__col_map(col) for col in df.columns]

    df = df.loc[:, [col for col in df.columns if col[0] not in 'VI']]

    for col in df.columns:
        if 'delta' in col:
            df.loc[:, col] = df.loc[:, col] * np.pi / 180

    return df

#主程序
if __name__ == '__main__':
    test()
