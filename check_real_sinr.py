"""
检查真实训练中的SINR值
"""
import numpy as np
import sys
sys.path.insert(0, '/home/qiankun/GAT_RA')
import Environment_marl_indoor
from arguments import get_args

args = get_args()

# 初始化环境（与训练相同）
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

# 使用相同的种子
np.random.seed(args.seed)
env.new_random_game()

print("=" * 80)
print("真实训练环境SINR检查")
print("=" * 80)

# 打印UAV位置
print("\nUAV位置:")
for i, v in enumerate(env.vehicles):
    print(f"  UAV {i}: ({v.position[0]:.1f}, {v.position[1]:.1f}, {v.position[2]:.1f})")

# 模拟一个episode的开始
env.renew_BS_channel()
env.renew_BS_channels_fastfading()

# 创建测试动作（所有UAV使用不同RB，最大功率）
actions = np.zeros((n_veh, 3))
for i in range(n_veh):
    actions[i, 0] = i % n_RB  # RB选择
    actions[i, 1] = 24.0  # 最大功率 24 dBm
    actions[i, 2] = 0.8  # rho压缩比

print("\n测试动作（所有UAV使用不同RB，最大功率24dBm）:")
for i in range(n_veh):
    print(f"  UAV {i}: RB={int(actions[i,0])}, Power={actions[i,1]:.1f}dBm, Rho={actions[i,2]:.2f}")

# 执行动作并获取性能指标
results = env.Compute_Performance_Reward_Train(actions, IS_PPO=True)
cellular_Rate, cellular_SINR, SE, EE, semantic_accuracy, semantic_EE, collisions = results

print("\n=" * 80)
print("性能指标")
print("=" * 80)

print("\nSINR (dB):")
for i in range(n_veh):
    sinr_dB = 10 * np.log10(cellular_SINR[i]) if cellular_SINR[i] > 0 else -np.inf
    success = "✅" if cellular_SINR[i] > 10**(3.16/10) else "❌"
    print(f"  UAV {i}: {sinr_dB:7.2f} dB (线性: {cellular_SINR[i]:.2e}) {success}")

print(f"\nSINR阈值:")
print(f"  成功: 3.16 dB (线性: {10**(3.16/10):.2f})")
print(f"  训练: 3.30 dB (线性: {10**(3.3/10):.2f})")

print("\n传输速率 (Mbps):")
for i in range(n_veh):
    print(f"  UAV {i}: {cellular_Rate[i]/1e6:.2f} Mbps")

print("\n语义准确度 (mAP):")
for i in range(n_veh):
    print(f"  UAV {i}: {semantic_accuracy[i]:.3f}")

print("\n语义能量效率:")
for i in range(n_veh):
    print(f"  UAV {i}: {semantic_EE[i]:.6f}")

print("\n碰撞状态:")
for i in range(n_veh):
    collision_status = "❌ 碰撞" if collisions[i] > 0 else "✅ 无碰撞"
    print(f"  UAV {i}: {collision_status}")

print("\n成功状态:")
for i in range(n_veh):
    success_status = "✅ 成功" if env.success[i] == 1 else "❌ 失败"
    print(f"  UAV {i}: {success_status}")

# 统计
print("\n=" * 80)
print("统计")
print("=" * 80)

num_success = np.sum(env.success)
avg_sinr_dB = 10 * np.log10(np.mean(cellular_SINR))
max_sinr_dB = 10 * np.log10(np.max(cellular_SINR))
min_sinr_dB = 10 * np.log10(np.min(cellular_SINR[cellular_SINR > 0])) if np.any(cellular_SINR > 0) else -np.inf

print(f"\n成功UAV数: {int(num_success)}/{n_veh}")
print(f"平均SINR: {avg_sinr_dB:.2f} dB")
print(f"最大SINR: {max_sinr_dB:.2f} dB")
print(f"最小SINR: {min_sinr_dB:.2f} dB")

# 分析：哪些UAV可以成功
print("\n=" * 80)
print("分析")
print("=" * 80)

can_succeed = cellular_SINR > 10**(3.16/10)
cannot_succeed = ~can_succeed

print(f"\n能成功的UAV: {np.where(can_succeed)[0]}")
print(f"不能成功的UAV: {np.where(cannot_succeed)[0]}")

if np.any(cannot_succeed):
    print("\n不能成功的UAV的SINR值:")
    for i in np.where(cannot_succeed)[0]:
        sinr_dB = 10 * np.log10(cellular_SINR[i]) if cellular_SINR[i] > 0 else -np.inf
        print(f"  UAV {i}: {sinr_dB:.2f} dB (需要 3.16 dB)")
        print(f"    差距: {3.16 - sinr_dB:.2f} dB")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if np.all(can_succeed):
    print("\n✅ 所有UAV都能成功（使用最大功率且无RB冲突）")
    print("   训练中的不均匀可能来自：")
    print("   1. 学习策略导致的RB冲突")
    print("   2. 功率选择不当")
    print("   3. 位置固定 + 联邦学习偏差")
else:
    num_failed = np.sum(cannot_succeed)
    print(f"\n❌ 有 {num_failed} 个UAV即使使用最大功率也无法成功！")
    print("   这些UAV的SINR低于阈值。")
    print("\n建议：")
    print("   1. 降低SINR阈值（3.16 → 2.5 dB）")
    print("   2. 每个episode重置位置")
    print("   3. 增加发射功率")

print("\n" + "=" * 80)

