#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析Semantic-EE为什么范围小
"""

import numpy as np
import math

# 参数
circuit_power = 0.06
semantic_A_max = 1.0
semantic_beta = 2.0

# 功率范围（dB）
power_levels_dB = [24, 21, 18, 15, 12, 9, 6, 3, 0]

# 压缩比范围
rho_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

# SINR范围（线性值，对应测试结果）
sinr_min_linear = 0.000796
sinr_max_linear = 1702.148
sinr_typical_linear = 0.086265  # 中位数

print("=" * 80)
print("Semantic-EE计算分析")
print("=" * 80)

def compute_semantic_accuracy(rho, sinr_linear):
    """计算语义准确度"""
    # 从代码中提取的公式
    sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
    # semantic_accuracy = A_max * (1 - exp(-beta * rho * SINR_linear))
    # 但实际代码可能不同，让我检查
    # 假设公式：A_max * (1 - exp(-beta * rho * SINR_linear))
    if sinr_linear > 0:
        accuracy = semantic_A_max * (1 - np.exp(-semantic_beta * rho * sinr_linear))
    else:
        accuracy = 0.0
    return accuracy

def compute_semantic_ee(accuracy, power_dB):
    """计算Semantic-EE"""
    power_linear = 10 ** (power_dB / 10)
    total_power = power_linear + circuit_power
    if total_power > 0:
        semantic_ee = accuracy / total_power
    else:
        semantic_ee = 0.0
    return semantic_ee

print("\n【理论分析】")
print("\n1. 功率对Semantic-EE的影响（假设准确度=1.0）:")
for power_dB in [0, 6, 12, 18, 24]:
    power_linear = 10 ** (power_dB / 10)
    total_power = power_linear + circuit_power
    semantic_ee = 1.0 / total_power
    print(f"  功率 {power_dB:2d} dB: power_linear={power_linear:8.2f}, "
          f"total_power={total_power:8.2f}, Semantic-EE={semantic_ee:.6f}")

print("\n2. 准确度对Semantic-EE的影响（功率=0 dB）:")
for accuracy in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    power_linear = 10 ** (0 / 10)  # 0 dB
    total_power = power_linear + circuit_power
    semantic_ee = accuracy / total_power
    print(f"  准确度 {accuracy:.1f}: Semantic-EE={semantic_ee:.6f}")

print("\n3. 实际场景分析（使用测试中的SINR值）:")
print("\n   情况1: 低SINR (0.086265线性，-10.66 dB), 不同功率和压缩比")
sinr_low = 0.086265
for power_dB in [0, 12, 24]:
    for rho in [0.1, 0.5, 0.9]:
        accuracy = compute_semantic_accuracy(rho, sinr_low)
        semantic_ee = compute_semantic_ee(accuracy, power_dB)
        print(f"  功率{power_dB:2d}dB, rho={rho:.1f}: "
              f"accuracy={accuracy:.6f}, Semantic-EE={semantic_ee:.6f}")

print("\n   情况2: 高SINR (1702.148线性，32.31 dB), 不同功率和压缩比")
sinr_high = 1702.148
for power_dB in [0, 12, 24]:
    for rho in [0.1, 0.5, 0.9]:
        accuracy = compute_semantic_accuracy(rho, sinr_high)
        semantic_ee = compute_semantic_ee(accuracy, power_dB)
        print(f"  功率{power_dB:2d}dB, rho={rho:.1f}: "
              f"accuracy={accuracy:.6f}, Semantic-EE={semantic_ee:.6f}")

print("\n4. 理论最大值分析:")
print("   如果准确度=1.0，功率=0 dB:")
max_ee = compute_semantic_ee(1.0, 0)
print(f"   Semantic-EE = {max_ee:.6f}")

print("\n   如果准确度=0.834701（测试中的最大值），功率=0 dB:")
max_ee = compute_semantic_ee(0.834701, 0)
print(f"   Semantic-EE = {max_ee:.6f}")

print("\n   如果准确度=0.834701，功率=6 dB:")
max_ee = compute_semantic_ee(0.834701, 6)
print(f"   Semantic-EE = {max_ee:.6f}")

print("\n   如果准确度=0.834701，功率=12 dB:")
max_ee = compute_semantic_ee(0.834701, 12)
print(f"   Semantic-EE = {max_ee:.6f}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
Semantic-EE = semantic_accuracy / (transmission_power + circuit_power)

影响Semantic-EE的因素：
1. 准确度（受SINR和压缩比影响）
2. 传输功率（线性值，10^(dB/10)）
3. 电路功率（0.06）

问题分析：
1. 功率值较大：24 dB = 251.19 (线性)，导致分母大
2. 准确度受SINR限制：大部分情况SINR低，准确度低
3. 压缩比影响准确度：rho小→准确度小

为什么范围小：
- 高功率时，即使准确度高，Semantic-EE也会被分母拉低
- 低SINR时，准确度本身就低
- 只有低功率+高准确度时，Semantic-EE才能达到较高值
""")

