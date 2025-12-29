#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比原始环境和当前环境的路径损耗计算
"""

import numpy as np
import math

print("=" * 80)
print("📊 信道模型对比分析")
print("=" * 80)

# 参数
fc = 6  # GHz
c = 3e8  # m/s

# 原始环境路径损耗公式
def path_loss_bs(position, bs_position=[12.5, 12.5], h_bs=5, h_ms=1.5):
    """原始环境：BSchannels路径损耗"""
    d1 = abs(position[0] - bs_position[0])
    d2 = abs(position[1] - bs_position[1])
    d_3d = math.sqrt(d1**2 + d2**2 + (h_bs - h_ms)**2)
    pl = 32.4 + 20 * math.log10(fc) + 31.9 * math.log10(d_3d)
    return pl

# 当前环境路径损耗公式
def path_loss_a2g(position, gbs_position=[12.5, 12.5, 0], eta_los=1.0, eta_nlos=20.0):
    """当前环境：A2GChannels路径损耗"""
    # 3D距离
    d_3d = math.sqrt((position[0] - gbs_position[0])**2 + 
                    (position[1] - gbs_position[1])**2 + 
                    (position[2] - gbs_position[2])**2)
    
    # 自由空间路径损耗
    fsp_loss = 20 * math.log10(d_3d) + 20 * math.log10(fc * 1e9) + 20 * math.log10(4 * math.pi / c)
    
    # LoS概率（简化，假设高仰角）
    d_2d = math.sqrt((position[0] - gbs_position[0])**2 + (position[1] - gbs_position[1])**2)
    h_uav = position[2]
    h_gbs = gbs_position[2]
    if d_2d > 0:
        theta = math.atan((h_uav - h_gbs) / d_2d) * 180 / math.pi
    else:
        theta = 90.0
    
    a, b = 9.61, 0.16
    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta - a)))
    
    # 期望路径损耗
    pl = fsp_loss + p_los * eta_los + (1 - p_los) * eta_nlos
    return pl, p_los

# 测试位置
test_positions = [
    ([12.5, 12.5, 1.5], "基站正上方(地面)"),
    ([12.5, 12.5, 50], "基站正上方(50m)"),
    ([12.5, 12.5, 100], "基站正上方(100m)"),
    ([12.5, 12.5, 150], "基站正上方(150m)"),
    ([12.5, 12.5, 200], "基站正上方(200m)"),
    ([25.0, 12.5, 100], "距离12.5m(100m高)"),
    ([0.0, 0.0, 100], "角落(100m高)"),
]

print("\n【路径损耗对比】")
print("-" * 80)
print(f"{'位置':<30} {'原始环境':<12} {'当前环境':<12} {'差异':<12} {'LoS概率':<10}")
print("-" * 80)

for pos, desc in test_positions:
    pl_bs = path_loss_bs(pos)
    pl_a2g, p_los = path_loss_a2g(pos)
    diff = pl_a2g - pl_bs
    print(f"{desc:<30} {pl_bs:>10.2f} dB {pl_a2g:>10.2f} dB {diff:>+10.2f} dB {p_los:>8.2%}")

# 分析问题
print("\n【问题分析】")
print("-" * 80)
print("原始环境公式: PL = 32.4 + 20*log10(fc) + 31.9*log10(d_3d)")
print("当前环境公式: PL = FSPL + p_los*eta_LoS + (1-p_los)*eta_NLoS")
print("其中: FSPL = 20*log10(d_3d) + 20*log10(fc*1e9) + 20*log10(4*pi/c)")

# 计算常数项差异
fsp_constant = 20 * math.log10(fc * 1e9) + 20 * math.log10(4 * math.pi / c)
bs_constant = 32.4 + 20 * math.log10(fc)
print(f"\n常数项对比:")
print(f"  原始环境: {bs_constant:.2f} dB")
print(f"  当前环境FSPL常数: {fsp_constant:.2f} dB")
print(f"  差异: {fsp_constant - bs_constant:.2f} dB")

# 计算距离项差异
d_test = 10  # 10m
pl_bs_d = 32.4 + 20 * math.log10(fc) + 31.9 * math.log10(d_test)
fsp_d = 20 * math.log10(d_test) + 20 * math.log10(fc * 1e9) + 20 * math.log10(4 * math.pi / c)
print(f"\n距离项对比 (d={d_test}m):")
print(f"  原始环境: {pl_bs_d:.2f} dB")
print(f"  当前环境FSPL: {fsp_d:.2f} dB")
print(f"  差异: {fsp_d - pl_bs_d:.2f} dB")

# 计算需要调整的参数
print("\n【调整建议】")
print("-" * 80)
print("为了匹配原始环境的SINR水平，需要:")
print("1. 降低路径损耗（减少eta_LoS和eta_NLoS）")
print("2. 或者调整FSPL公式使其接近原始环境")
print("3. 或者添加补偿项")

# 计算补偿值
pos_test = [12.5, 12.5, 100]  # 典型UAV位置
pl_bs_test = path_loss_bs(pos_test)
pl_a2g_test, _ = path_loss_a2g(pos_test)
compensation = pl_bs_test - pl_a2g_test
print(f"\n典型位置补偿值 (100m高): {compensation:.2f} dB")

