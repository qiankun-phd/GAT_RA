#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试SINR计算
"""

import numpy as np
import sys
import os
from arguments import get_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Environment_marl_indoor import Environ

# 创建环境
args = get_args()
n_veh = 6
n_RB = 10

env = Environ(n_veh=n_veh, n_RB=n_RB, 
              optimization_target='EE',
              beta=0.5, circuit_power=0.06)

env.new_random_game()
env.renew_BS_channel()
env.renew_BS_channels_fastfading()

# 打印信道信息
print("=" * 80)
print("信道信息")
print("=" * 80)
print(f"路径损耗范围: {np.min(env.cellular_pathloss):.2f} ~ {np.max(env.cellular_pathloss):.2f} dB")
print(f"阴影衰落范围: {np.min(env.cellular_Shadowing):.2f} ~ {np.max(env.cellular_Shadowing):.2f} dB")
print(f"信道增益范围: {np.min(env.cellular_channels_abs):.2f} ~ {np.max(env.cellular_channels_abs):.2f} dB")
print(f"快衰落信道范围: {np.min(env.cellular_channels_with_fastfading):.2f} ~ {np.max(env.cellular_channels_with_fastfading):.2f} dB")

# 测试一个简单场景
actions_all = np.zeros([n_veh, 3], dtype='float32')
for i in range(n_veh):
    actions_all[i, 0] = i % n_RB  # 不同RB
    actions_all[i, 1] = 24  # 最大功率
    actions_all[i, 2] = 0.5

# 手动计算信号
print("\n" + "=" * 80)
print("手动计算信号（UAV 0）")
print("=" * 80)
uav_idx = 0
rb_idx = int(actions_all[uav_idx, 0])
power_dB = actions_all[uav_idx, 1]

channel_gain_dB = env.cellular_channels_with_fastfading[uav_idx, rb_idx]
signal_linear = 10 ** ((power_dB - channel_gain_dB + env.vehAntGain + env.bsAntGain - env.bsNoiseFigure) / 10)

print(f"UAV {uav_idx} 位置: {env.vehicles[uav_idx].position}")
print(f"选择的RB: {rb_idx}")
print(f"功率: {power_dB} dB")
print(f"信道增益: {channel_gain_dB:.2f} dB")
print(f"信号（线性）: {signal_linear:.6f}")
print(f"信号（dB）: {10*np.log10(signal_linear):.2f} dB")

# 计算干扰
interference = env.sig2[rb_idx]  # 噪声
for i in range(n_veh):
    if i != uav_idx and int(actions_all[i, 0]) == rb_idx:
        other_power_dB = actions_all[i, 1]
        other_channel_dB = env.cellular_channels_with_fastfading[i, rb_idx]
        other_signal = 10 ** ((other_power_dB - other_channel_dB + env.vehAntGain + env.bsAntGain - env.bsNoiseFigure) / 10)
        interference += other_signal
        print(f"  来自UAV {i}的干扰: {other_signal:.6f}")

print(f"总干扰+噪声: {interference:.6f}")
print(f"噪声功率: {env.sig2[rb_idx]:.6f}")

# 计算SINR
sinr_linear = signal_linear / interference
sinr_dB = 10 * np.log10(sinr_linear)
print(f"SINR（线性）: {sinr_linear:.6f}")
print(f"SINR（dB）: {sinr_dB:.2f} dB")

# 使用环境函数计算
results = env.Compute_Performance_Reward_Train(actions_all, IS_PPO=True)
(cellular_Rate, cellular_SINR, SE, EE, 
 semantic_accuracy, semantic_EE, collisions) = results

print(f"\n环境计算的SINR（UAV 0）: {cellular_SINR[0]:.6f} (线性), {10*np.log10(cellular_SINR[0]):.2f} dB")

