#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试调整后的A2G信道模型
"""

import numpy as np
import sys
import os
from arguments import get_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Environment_marl_indoor import Environ

def test_adjusted_sinr():
    """测试调整后的SINR"""
    print("=" * 80)
    print("📊 调整后环境SINR测试")
    print("=" * 80)
    
    args = get_args()
    n_veh = 6
    n_RB = 10
    
    env = Environ(n_veh=n_veh, n_RB=n_RB, 
                  optimization_target='EE',
                  beta=0.5, circuit_power=0.06)
    
    env.new_random_game()
    env.renew_BS_channel()
    env.renew_BS_channels_fastfading()
    
    power_levels_dB = [24, 18, 12, 6, 0]
    
    sinr_results = []
    
    for test_round in range(10):
        env.new_random_game()
        env.renew_BS_channel()
        env.renew_BS_channels_fastfading()
        
        for power_dB in power_levels_dB:
            actions_all = np.zeros([n_veh, 3], dtype='float32')
            for i in range(n_veh):
                actions_all[i, 0] = i % n_RB
                actions_all[i, 1] = power_dB
                actions_all[i, 2] = 0.5
            
            results = env.Compute_Performance_Reward_Train(actions_all, IS_PPO=True)
            (cellular_Rate, cellular_SINR, SE, EE, 
             semantic_accuracy, semantic_EE, collisions) = results
            
            for i in range(n_veh):
                sinr_linear = cellular_SINR[i]
                sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
                sinr_results.append({
                    'power_dB': power_dB,
                    'sinr_dB': sinr_dB,
                    'success': env.success[i]
                })
    
    sinr_dB_all = [r['sinr_dB'] for r in sinr_results if r['sinr_dB'] != -np.inf]
    
    print(f"\n【SINR统计（dB）】")
    print(f"  样本数: {len(sinr_dB_all)}")
    if sinr_dB_all:
        print(f"  最小值: {np.min(sinr_dB_all):.2f} dB")
        print(f"  最大值: {np.max(sinr_dB_all):.2f} dB")
        print(f"  平均值: {np.mean(sinr_dB_all):.2f} dB")
        print(f"  中位数: {np.median(sinr_dB_all):.2f} dB")
        print(f"  标准差: {np.std(sinr_dB_all):.2f} dB")
    
    # 阈值检查
    print(f"\n【阈值检查】")
    thresholds = [2.5, 3.3, 3.16]
    for threshold_dB in thresholds:
        above_threshold = sum(1 for r in sinr_results 
                             if r['sinr_dB'] != -np.inf and r['sinr_dB'] > threshold_dB)
        total = len(sinr_results)
        percentage = (above_threshold / total * 100) if total > 0 else 0
        print(f"  SINR > {threshold_dB:.2f} dB: {above_threshold}/{total} ({percentage:.2f}%)")
    
    # 按功率统计
    print(f"\n【按功率级别统计SINR（dB）】")
    for power_dB in power_levels_dB:
        power_sinr = [r['sinr_dB'] for r in sinr_results 
                     if r['power_dB'] == power_dB and r['sinr_dB'] != -np.inf]
        if power_sinr:
            print(f"  功率 {power_dB:2d} dB: "
                  f"min={np.min(power_sinr):6.2f} dB, "
                  f"max={np.max(power_sinr):6.2f} dB, "
                  f"mean={np.mean(power_sinr):6.2f} dB")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_adjusted_sinr()

