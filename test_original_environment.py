#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试原始环境中的SINR和EE范围
"""

import numpy as np
import sys
import os

# 添加原始环境路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'origin'))

# 导入原始环境
from Environment_marl_indoor import Environ
from arguments import get_args

def test_original_sinr_ee():
    """测试原始环境的SINR和EE"""
    print("=" * 80)
    print("📊 原始环境SINR和EE范围测试")
    print("=" * 80)
    
    args = get_args()
    n_veh = 6
    n_RB = 10
    
    # 创建原始环境
    env = Environ(n_veh=n_veh, n_RB=n_RB, 
                  optimization_target='SE_EE',
                  beta=0.5, circuit_power=0.06)
    
    # 初始化环境
    env.new_random_game()
    env.renew_BS_channel()
    env.renew_BS_channels_fastfading()
    
    # 测试不同功率级别
    power_levels_dB = [24, 21, 18, 15, 12, 9, 6, 3, 0]
    
    sinr_results = []
    ee_results = []
    
    # 测试多个随机位置
    for test_round in range(10):
        env.new_random_game()
        env.renew_BS_channel()
        env.renew_BS_channels_fastfading()
        
        for power_idx, power_dB in enumerate(power_levels_dB):
            # 创建测试动作：每个UAV选择不同的RB，使用相同功率
            actions_all = np.zeros([n_veh, 2], dtype='float32')
            for i in range(n_veh):
                actions_all[i, 0] = i % n_RB  # 不同RB避免碰撞
                actions_all[i, 1] = power_dB  # 功率（PPO模式，直接dB值）
            
            # 计算性能（使用原始环境的函数）
            results = env.Compute_Performance_Reward_Train(actions_all, IS_PPO=True)
            (cellular_Rate, cellular_SINR, SE, EE) = results
            
            # 记录SINR和EE
            for i in range(n_veh):
                sinr_linear = cellular_SINR[i]
                sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
                sinr_results.append({
                    'test_round': test_round,
                    'uav': i,
                    'power_dB': power_dB,
                    'sinr_linear': sinr_linear,
                    'sinr_dB': sinr_dB,
                    'success': env.success[i],
                    'rate': cellular_Rate[i],
                    'SE': SE[i],
                    'EE': EE[i]
                })
                
                if EE[i] > 0:
                    ee_results.append({
                        'test_round': test_round,
                        'uav': i,
                        'power_dB': power_dB,
                        'sinr_linear': sinr_linear,
                        'sinr_dB': sinr_dB,
                        'rate': cellular_Rate[i],
                        'SE': SE[i],
                        'EE': EE[i],
                        'success': env.success[i]
                    })
    
    # SINR统计分析
    sinr_linear_all = [r['sinr_linear'] for r in sinr_results if r['sinr_linear'] > 0]
    sinr_dB_all = [r['sinr_dB'] for r in sinr_results if r['sinr_dB'] != -np.inf]
    
    print(f"\n【SINR统计（线性值）】")
    print(f"  样本数: {len(sinr_linear_all)}")
    if sinr_linear_all:
        print(f"  最小值: {np.min(sinr_linear_all):.6f}")
        print(f"  最大值: {np.max(sinr_linear_all):.6f}")
        print(f"  平均值: {np.mean(sinr_linear_all):.6f}")
        print(f"  中位数: {np.median(sinr_linear_all):.6f}")
        print(f"  标准差: {np.std(sinr_linear_all):.6f}")
    
    print(f"\n【SINR统计（dB值）】")
    print(f"  样本数: {len(sinr_dB_all)}")
    if sinr_dB_all:
        print(f"  最小值: {np.min(sinr_dB_all):.2f} dB")
        print(f"  最大值: {np.max(sinr_dB_all):.2f} dB")
        print(f"  平均值: {np.mean(sinr_dB_all):.2f} dB")
        print(f"  中位数: {np.median(sinr_dB_all):.2f} dB")
        print(f"  标准差: {np.std(sinr_dB_all):.2f} dB")
    
    # 按功率级别统计SINR
    print(f"\n【按功率级别统计SINR（dB）】")
    for power_dB in power_levels_dB:
        power_sinr = [r['sinr_dB'] for r in sinr_results 
                     if r['power_dB'] == power_dB and r['sinr_dB'] != -np.inf]
        if power_sinr:
            print(f"  功率 {power_dB:2d} dB: "
                  f"min={np.min(power_sinr):6.2f} dB, "
                  f"max={np.max(power_sinr):6.2f} dB, "
                  f"mean={np.mean(power_sinr):6.2f} dB")
    
    # 阈值检查
    print(f"\n【阈值检查】")
    thresholds = [2.5, 3.3, 3.16]  # dB
    for threshold_dB in thresholds:
        threshold_linear = 10 ** (threshold_dB / 10)
        above_threshold = sum(1 for r in sinr_results 
                             if r['sinr_linear'] > threshold_linear)
        total = len(sinr_results)
        percentage = (above_threshold / total * 100) if total > 0 else 0
        print(f"  SINR > {threshold_dB:.2f} dB ({threshold_linear:.4f} linear): "
              f"{above_threshold}/{total} ({percentage:.2f}%)")
    
    # EE统计分析
    ee_all = [r['EE'] for r in ee_results if r['EE'] > 0]
    
    print(f"\n" + "=" * 80)
    print("📊 EE (Energy Efficiency) 统计")
    print("=" * 80)
    
    print(f"\n【EE统计】")
    print(f"  样本数: {len(ee_all)}")
    if ee_all:
        print(f"  最小值: {np.min(ee_all):.6f}")
        print(f"  最大值: {np.max(ee_all):.6f}")
        print(f"  平均值: {np.mean(ee_all):.6f}")
        print(f"  中位数: {np.median(ee_all):.6f}")
        print(f"  标准差: {np.std(ee_all):.6f}")
    
    # 按功率级别统计EE
    print(f"\n【按功率级别统计EE】")
    for power_dB in power_levels_dB:
        power_ee = [r['EE'] for r in ee_results 
                   if r['power_dB'] == power_dB and r['EE'] > 0]
        if power_ee:
            print(f"  功率 {power_dB:2d} dB: "
                  f"min={np.min(power_ee):.6f}, "
                  f"max={np.max(power_ee):.6f}, "
                  f"mean={np.mean(power_ee):.6f}")
    
    # 成功情况下的EE
    success_ee = [r['EE'] for r in ee_results 
                 if r['success'] == 1 and r['EE'] > 0]
    print(f"\n【成功情况下的EE】")
    print(f"  样本数: {len(success_ee)}")
    if success_ee:
        print(f"  最小值: {np.min(success_ee):.6f}")
        print(f"  最大值: {np.max(success_ee):.6f}")
        print(f"  平均值: {np.mean(success_ee):.6f}")
        print(f"  中位数: {np.median(success_ee):.6f}")
    
    # SE统计分析
    se_all = [r['SE'] for r in sinr_results if r['SE'] > 0]
    print(f"\n【SE (Spectral Efficiency) 统计】")
    print(f"  样本数: {len(se_all)}")
    if se_all:
        print(f"  最小值: {np.min(se_all):.6f}")
        print(f"  最大值: {np.max(se_all):.6f}")
        print(f"  平均值: {np.mean(se_all):.6f}")
        print(f"  中位数: {np.median(se_all):.6f}")
    
    # Rate统计分析
    rate_all = [r['rate'] for r in sinr_results if r['rate'] > 0]
    print(f"\n【Rate (传输速率) 统计】")
    print(f"  样本数: {len(rate_all)}")
    if rate_all:
        print(f"  最小值: {np.min(rate_all):.6f} Mbps")
        print(f"  最大值: {np.max(rate_all):.6f} Mbps")
        print(f"  平均值: {np.mean(rate_all):.6f} Mbps")
        print(f"  中位数: {np.median(rate_all):.6f} Mbps")
    
    # 理论EE分析
    print(f"\n" + "=" * 80)
    print("📐 理论EE分析")
    print("=" * 80)
    
    print("\n【理论EE计算（假设Rate=1 Mbps）】")
    for power_dB in [0, 6, 12, 18, 24]:
        power_linear = 10 ** (power_dB / 10)
        total_power = power_linear + 0.06  # circuit_power = 0.06
        ee_theoretical = 1.0 / total_power  # Rate = 1 Mbps
        print(f"  功率 {power_dB:2d} dB: power_linear={power_linear:8.2f}, "
              f"total_power={total_power:8.2f}, EE={ee_theoretical:.6f}")
    
    return sinr_results, ee_results


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🔬 原始环境指标测试")
    print("=" * 80)
    print("\n测试配置:")
    print("  - UAV数量: 6")
    print("  - RB数量: 10")
    print("  - 测试轮数: 10 (随机位置)")
    print("  - 功率范围: 0-24 dB")
    print("  - 电路功率: 0.06")
    print("\n")
    
    # 测试SINR和EE
    sinr_results, ee_results = test_original_sinr_ee()
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    # SINR总结
    sinr_linear_all = [r['sinr_linear'] for r in sinr_results if r['sinr_linear'] > 0]
    if sinr_linear_all:
        sinr_dB_all = [10 * np.log10(s) for s in sinr_linear_all]
        print(f"\n【SINR范围】")
        print(f"  范围: {np.min(sinr_dB_all):.2f} dB ~ {np.max(sinr_dB_all):.2f} dB")
        print(f"  平均值: {np.mean(sinr_dB_all):.2f} dB")
        print(f"  中位数: {np.median(sinr_dB_all):.2f} dB")
    
    # EE总结
    ee_all = [r['EE'] for r in ee_results if r['EE'] > 0]
    if ee_all:
        print(f"\n【EE范围】")
        print(f"  范围: {np.min(ee_all):.6f} ~ {np.max(ee_all):.6f}")
        print(f"  平均值: {np.mean(ee_all):.6f}")
        print(f"  中位数: {np.median(ee_all):.6f}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

