import json
import numpy as np
from interface_v3 import GridFunc

def load_case_from_json(json_file, case_id):
    # 从json文件中读取数据
    with open(json_file, 'r') as f:
        cases = json.load(f)
    
    # 查找对应ID的工况数据
    for case in cases:
        if case['id'] == case_id:
            gts = case['gts']
            dat = case['dat']
            break
    else:
        raise ValueError(f"未找到ID为{case_id}的工况数据")
    
    # 设置发电机的有功功率
    gen_p = np.array(list(map(float, gts['fd'])), dtype=float)
    
    # 设置母线负荷
    fh = list(map(float, gts['fh']))
    load_p = np.array(fh[::2], dtype=float)  # 取fh中的奇数项（有功功率P）
    load_q = np.array(fh[1::2], dtype=float) # 取fh中的偶数项（无功功率Q）
    
    # 设置故障参数
    bus = int(gts['position'][0])
    x = float(gts['gzjddzcs'][1])
    fault_begin_time = float(gts['gzsj'][1])
    fault_end_time = float(gts['gzsj'][2])
    
    # 设置Efd和Pt参数
    Efd = np.array([dat['Efd'][0][0], dat['Efd'][1][0], dat['Efd'][2][0]], dtype=float)
    Pt = np.array([dat['Tm'][0][0], dat['Tm'][1][0], dat['Tm'][2][0]], dtype=float)
    
    # 设置delta参数
    delta = np.array([dat['delta'][0][0], dat['delta'][1][0], dat['delta'][2][0]], dtype=float)
    
    # 设置Ed参数
    Ed = np.array([
        dat['Edp'][0][0],          # 发电机1的Ed'
        dat['Edpp'][1][0],         # 发电机2的Ed''
        dat['Edpp'][2][0]          # 发电机3的Ed''
    ], dtype=float)
    
    # 设置Eq参数
    Eq = np.array([
        dat['Eqp'][0][0],          # 发电机1的Eq'
        dat['Eqpp'][1][0],         # 发电机2的Eq''
        dat['Eqpp'][2][0]          # 发电机3的Eq''
    ], dtype=float)
    
    return {
        'Efd': Efd,
        'Pt': Pt,
        'gen_p': gen_p,
        'load_p': load_p,
        'load_q': load_q,
        'bus': bus,
        'x': x,
        'fault_begin_time': fault_begin_time,
        'fault_end_time': fault_end_time,
        'delta': delta,
        'Ed': Ed,
        'Eq': Eq
    }

if __name__ == '__main__':
    json_file = 'merged_output_ztdata_t_modi_0311.json'  # json文件名
    case_id = int(2)
    
    try:
        case_data = load_case_from_json(json_file, case_id)
        
        # 直接输出读取的参数以验证正确性
        #print("\n验证读取的json文件参数:")
        #print("Efd:", case_data['Efd'])
        #print("Pt:", case_data['Pt'])
        #print("gen_p:", case_data['gen_p'])
        #print("load_p:", case_data['load_p'])
        #print("load_q:", case_data['load_q'])
        #print("bus:", case_data['bus'])
        #print("x:", case_data['x'])
        #print("fault_begin_time:", case_data['fault_begin_time'])
        #print("fault_end_time:", case_data['fault_end_time'])
        #print("delta:", case_data['delta'])
        #print("Ed:", case_data['Ed'])
        #print("Eq:", case_data['Eq'])
        
        # 设置发电机参数(所有工况适用)
        Xd_ = np.array([0.0608, 0.1198, 0.1813], dtype=float)    # 发电机1, 2, 3对应的 Xd'
        Xd__ = np.array([0.0608, 0.1198, 0.1813], dtype=float)   # 发电机1, 2, 3对应的 Xd''
        Xq__ = np.array([0.0969, 0.1969, 0.25], dtype=float)     # 发电机1, 2, 3对应的 Xq''
        
        Xd = np.array([0.146, 0.8958, 1.3125], dtype=float)      # 发电机1, 2, 3对应的 Xd
        Xq = np.array([0.0969, 0.8645, 1.2578], dtype=float)     # 发电机1, 2, 3对应的 Xq
        Xq_ = np.array([0.0969, 0.1969, 0.25], dtype=float)      # 发电机1, 2, 3对应的 Xq'
        Td0_ = np.array([8.96, 6.0, 5.89], dtype=float)          # 发电机1, 2, 3对应的 Td0'
        Td0__ = np.array([6.0, 6.0, 6.0], dtype=float)           # 发电机1, 2, 3对应的 Td0''
        Tq0_ = np.array([0.0, 0.535, 0.6], dtype=float)          # 发电机1, 2, 3对应的 Tq0'
        Tq0__ = np.array([0.535, 0.535, 0.535], dtype=float)     # 发电机1, 2, 3对应的 Tq0''
        Tj = np.array([47.28, 12.8, 6.02], dtype=float)          # 发电机1, 2, 3对应的 Tj
        
        # 其他参数设置（同一工况相同，但不同工况不同）
        Efd = case_data['Efd'] # 发电机1, 2, 3对应的 Efd（励磁电压）
        Pt = case_data['Pt']      # 发电机1, 2, 3对应的 Pt（机械功率）
        
        # 设置发电机的有功功率（同一工况相同，但不同工况不同）
        gen_p = case_data['gen_p']       # 发电机1, 2, 3对应的有功功率, 标幺值
        
        # 设置母线负荷（同一工况相同，但不同工况不同）
        load_p = case_data['load_p']  # 母线5, 6, 8对应的有功负荷P, 标幺值
        load_q = case_data['load_q']    # 母线5, 6, 8对应的无功负荷Q, 标幺值

        # 设置故障参数（同一工况相同，但不同工况不同）
        bus = case_data['bus']               # 故障母线号
        x = case_data['x']           # 接地电抗, 标幺值
        repair_time = case_data['fault_end_time']  # 故障修复时间，对应fault_end_time
        
        # 创建GridFunc对象
        grid_func = GridFunc(Xd_, Xd__, Xq__, gen_p, load_p, load_q, bus, x, repair_time)
        
        # 设置时间推进参数
        time_step = 0.01          # 时间步长为10ms
        total_time = 10            # 总时间为10
        max_iterations = 50        # 最大迭代步设置为50
        residual_threshold = 1e-3  # 残差设定为10^-3
        
        # 初始化时间
        time = 0
        
        # 设置初始的发电机功角、电势
        delta = case_data['delta']  # 发电机1, 2, 3对应的功角（弧度制）
        Ed = case_data['Ed']  # 发电机1的Ed', 发电机2, 3的Ed''
        Eq = case_data['Eq']  # 发电机1的Eq', 发电机2, 3的Eq''
        
        # 用于存储每次时间步的结果
        results = []
        
        # 外层时间推进循环
        while time < total_time + time_step:
            # 当时间小于故障开始时间时，参数保持不变，不进入内层迭代
            if time < case_data['fault_begin_time']:
                # 保存当前时间步的结果（参数保持不变）
                results.append((time, delta.copy(), Ed.copy(), Eq.copy()))
                
                # 更新时间
                time += time_step
                continue
            
            # 内层迭代循环
            for iteration in range(max_iterations):
                # 调用get_result方法获取结果
                Id, Iq, Vd, Vq, psi_d, psi_q = grid_func.get_result(delta, Ed, Eq, time)
                
                # 计算残差（这里以Vd和Vq的变化作为残差的示例）
                if iteration > 0:
                    residual = np.max(np.abs(Vd - prev_Vd)) + np.max(np.abs(Vq - prev_Vq))
                    # 检查残差是否满足条件
                    if residual < residual_threshold:
                        break
                
                # 保存当前Vd和Vq用于下一次迭代比较
                prev_Vd, prev_Vq = Vd.copy(), Vq.copy()
                
                #=========================================================================
                #利用ODE求解器更新delta、Ed、Eq等参数，这里仅为示例
                #输入：前面已经调用get_result获得了Id, Iq, Vd, Vq, psi_d, psi_q，其他常参数也已经定义
                #输出：正常应当输出几个数组：delta、omega、Ed1，Ed2，Eq1，Eq2，可以参考get_result函数return以上参数
                #假设ODE接口函数为DeepGrid_ODE,可以通过以下形式传参：
                
                delta, omega, Ed1, Ed2, Eq1, Eq2 = DeepGrid_ODE(Id, Iq, Vd, Vq, psi_d, psi_q)  #仅举例，输入参数根据需要删减
                
                #=========================================================================
                
                #进一步定义0，6，6机型对应的Ed和Eq
                Ed = np.array([Ed1[0], Ed2[1], Ed2[2]], dtype=float)
                Eq = np.array([Eq1[0], Eq2[1], Eq2[2]], dtype=float)
                
            # 保存当前时间步的结果
            results.append((time, Id, Iq, Vd, Vq, psi_d, psi_q))
            
            # 更新时间
            time += time_step
    
    except Exception as e:
        print(f"发生错误: {e}")