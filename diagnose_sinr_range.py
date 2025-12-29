"""
诊断脚本：计算每个UAV在固定位置下的SINR范围
"""
import numpy as np
import sys
sys.path.insert(0, '/home/qiankun/GAT_RA')
import Environment_marl_indoor
from arguments import get_args

args = get_args()

# 初始化环境
n_veh = 6
n_RB = 10
env = Environment_marl_indoor.Environ(
    n_veh, n_RB,
    beta=0.5,
    circuit_power=0.06,
    optimization_target='EE',
    semantic_A_max=1.0,
    semantic_beta=2.0
)

# 初始化游戏（固定种子）
np.random.seed(args.seed)
env.new_random_game()
env.renew_BS_channel()
env.renew_BS_channels_fastfading()

print("=" * 80)
print("UAV位置与SINR范围诊断")
print("=" * 80)

# 获取基站位置（室内环境，假设基站在中心）
# 基于代码，基站在区域中心地面
GBS_pos = np.array([env.width/2, env.height/2, 0])
print(f"\n基站位置 (推测): {GBS_pos}")
print(f"区域大小: {env.width}m x {env.height}m")
print(f"UAV高度范围: {env.height_min}m - {env.height_max}m")

# 获取功率范围
power_dB_list = env.cellular_power_dB_List
power_min_dB = min(power_dB_list)
power_max_dB = max(power_dB_list)
power_min_linear = 10 ** (power_min_dB / 10)
power_max_linear = 10 ** (power_max_dB / 10)

print(f"\n功率范围:")
print(f"  最小: {power_min_dB:.2f} dBm ({power_min_linear:.6f} W)")
print(f"  最大: {power_max_dB:.2f} dBm ({power_max_linear:.6f} W)")
print(f"  可选功率: {power_dB_list} dBm")

# 噪声功率
noise_power = env.sig2[0]  # 假设所有RB噪声相同
print(f"\n噪声功率: {10*np.log10(noise_power):.2f} dBm ({noise_power:.2e} W)")

# SINR阈值
sinr_threshold_linear = 10 ** (3.16 / 10)
training_sinr_threshold = 3.3
print(f"\nSINR阈值:")
print(f"  成功判断: 3.16 dB (线性: {sinr_threshold_linear:.2f})")
print(f"  训练奖励: {training_sinr_threshold} dB (线性: {10**(training_sinr_threshold/10):.2f})")

print("\n" + "=" * 80)
print("各UAV详细信息")
print("=" * 80)

for i, vehicle in enumerate(env.vehicles):
    pos = vehicle.position
    
    # 计算到基站的3D距离
    distance_3d = np.linalg.norm(pos - GBS_pos)
    distance_2d = np.linalg.norm(pos[:2] - GBS_pos[:2])
    
    print(f"\n【UAV {i}】")
    print(f"  位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m")
    print(f"  到基站距离: 2D={distance_2d:.1f}m, 3D={distance_3d:.1f}m")
    
    # 获取信道增益（路径损耗，dB为负值）
    # 注意：cellular_channels_abs是路径损耗（负值），不是增益
    if isinstance(env.cellular_channels_abs, np.ndarray) and env.cellular_channels_abs.ndim > 1:
        path_loss_dB = float(env.cellular_channels_abs[i, 0])  # 第i个UAV的第一个RB
    else:
        path_loss_dB = float(env.cellular_channels_abs[i])
    channel_gain_linear = 10 ** (path_loss_dB / 10)  # 路径损耗是负值，所以增益<1
    
    print(f"  路径损耗: {path_loss_dB:.2f} dB (信道增益线性: {channel_gain_linear:.2e})")
    
    # 计算接收信号功率范围
    received_power_min = power_min_linear * channel_gain_linear
    received_power_max = power_max_linear * channel_gain_linear
    
    print(f"  接收功率范围:")
    print(f"    最小: {10*np.log10(received_power_min):.2f} dBm ({received_power_min:.2e} W)")
    print(f"    最大: {10*np.log10(received_power_max):.2f} dBm ({received_power_max:.2e} W)")
    
    # 计算SINR范围（假设没有干扰，最好情况）
    # SINR = 接收信号功率 / 噪声功率
    sinr_min_linear = received_power_min / noise_power
    sinr_max_linear = received_power_max / noise_power
    
    sinr_min_dB = 10 * np.log10(sinr_min_linear)
    sinr_max_dB = 10 * np.log10(sinr_max_linear)
    
    print(f"  SINR范围 (无干扰):")
    print(f"    最小功率: {sinr_min_dB:.2f} dB (线性: {sinr_min_linear:.2f})")
    print(f"    最大功率: {sinr_max_dB:.2f} dB (线性: {sinr_max_linear:.2f})")
    
    # 判断能否达到阈值
    can_succeed_min = sinr_min_linear > sinr_threshold_linear
    can_succeed_max = sinr_max_linear > sinr_threshold_linear
    can_train_min = sinr_min_dB > training_sinr_threshold
    can_train_max = sinr_max_dB > training_sinr_threshold
    
    print(f"  能否成功:")
    print(f"    最小功率: {'✅ 是' if can_succeed_min else '❌ 否'} (成功阈值3.16dB)")
    print(f"    最大功率: {'✅ 是' if can_succeed_max else '❌ 否'}")
    print(f"  能否获得训练奖励:")
    print(f"    最小功率: {'✅ 是' if can_train_min else '❌ 否'} (训练阈值3.3dB)")
    print(f"    最大功率: {'✅ 是' if can_train_max else '❌ 否'}")
    
    # 计算需要的最小功率
    required_power_linear_succeed = (sinr_threshold_linear * noise_power) / channel_gain_linear
    required_power_dBm_succeed = 10 * np.log10(required_power_linear_succeed)
    
    required_power_linear_train = (10**(training_sinr_threshold/10) * noise_power) / channel_gain_linear
    required_power_dBm_train = 10 * np.log10(required_power_linear_train)
    
    print(f"  需要的最小发射功率:")
    print(f"    达到成功阈值: {required_power_dBm_succeed:.2f} dBm ({'✅ 可达' if required_power_dBm_succeed <= power_max_dB else '❌ 超限'})")
    print(f"    达到训练阈值: {required_power_dBm_train:.2f} dBm ({'✅ 可达' if required_power_dBm_train <= power_max_dB else '❌ 超限'})")

print("\n" + "=" * 80)
print("汇总统计")
print("=" * 80)

# 统计能成功的UAV数量
can_succeed_count = 0
can_train_count = 0

for i, vehicle in enumerate(env.vehicles):
    if isinstance(env.cellular_channels_abs, np.ndarray) and env.cellular_channels_abs.ndim > 1:
        path_loss_dB = float(env.cellular_channels_abs[i, 0])
    else:
        path_loss_dB = float(env.cellular_channels_abs[i])
    channel_gain_linear = 10 ** (path_loss_dB / 10)
    received_power_max = power_max_linear * channel_gain_linear
    sinr_max_linear = received_power_max / noise_power
    sinr_max_dB = 10 * np.log10(sinr_max_linear)
    
    if sinr_max_linear > sinr_threshold_linear:
        can_succeed_count += 1
    if sinr_max_dB > training_sinr_threshold:
        can_train_count += 1

print(f"\n使用最大功率时:")
print(f"  能达到成功阈值(3.16dB)的UAV数: {can_succeed_count}/{n_veh}")
print(f"  能达到训练阈值(3.3dB)的UAV数: {can_train_count}/{n_veh}")
print(f"  完全无法成功的UAV数: {n_veh - can_succeed_count}/{n_veh}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if can_succeed_count < n_veh:
    print(f"\n⚠️  有 {n_veh - can_succeed_count} 个UAV即使使用最大功率也无法达到SINR阈值！")
    print(f"   这解释了为什么这些UAV成功率为0%。")
    print(f"\n建议:")
    print(f"  1. 每个episode重新随机化UAV位置")
    print(f"  2. 降低SINR阈值 (3.16 → 2.8 dB)")
    print(f"  3. 增加最大发射功率")
else:
    print(f"\n✅ 所有UAV都能达到SINR阈值（使用最大功率）")
    print(f"   位置不均匀的问题可能来自干扰或其他因素。")

print("\n" + "=" * 80)

