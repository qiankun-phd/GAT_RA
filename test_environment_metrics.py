#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试环境指标：SINR范围和Semantic-EE范围
"""

import numpy as np
import sys
import os
from arguments import get_args

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Environment_marl_indoor import Environ

def test_sinr_range():
    """测试SINR范围"""
    print("=" * 80)
    print("📊 SINR范围测试")
    print("=" * 80)
    
    args = get_args()
    n_veh = 6
    n_RB = 10
    
    env = Environ(n_veh=n_veh, n_RB=n_RB, 
                  optimization_target='EE',
                  beta=0.5, circuit_power=0.06)
    
    # 初始化环境
    env.new_random_game()
    env.renew_BS_channel()
    env.renew_BS_channels_fastfading()
    
    # 测试不同功率级别
    power_levels_dB = [24, 21, 18, 15, 12, 9, 6, 3, 0]
    
    sinr_results = []
    
    # 测试多个随机位置
    for test_round in range(10):
        env.new_random_game()
        env.renew_BS_channel()
        env.renew_BS_channels_fastfading()
        
        for power_idx, power_dB in enumerate(power_levels_dB):
            # 创建测试动作：每个UAV选择不同的RB，使用相同功率
            actions_all = np.zeros([n_veh, 3], dtype='float32')
            for i in range(n_veh):
                actions_all[i, 0] = i % n_RB  # 不同RB避免碰撞
                actions_all[i, 1] = power_dB  # 功率
                actions_all[i, 2] = 0.5  # 压缩比
            
            # 计算性能
            results = env.Compute_Performance_Reward_Train(actions_all, IS_PPO=True)
            (cellular_Rate, cellular_SINR, SE, EE, 
             semantic_accuracy, semantic_EE, collisions) = results
            
            # 记录SINR（线性值和dB值）
            for i in range(n_veh):
                sinr_linear = cellular_SINR[i]
                sinr_dB = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -np.inf
                sinr_results.append({
                    'test_round': test_round,
                    'uav': i,
                    'power_dB': power_dB,
                    'sinr_linear': sinr_linear,
                    'sinr_dB': sinr_dB,
                    'success': env.success[i]
                })
    
    # 统计分析
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
    
    # 按功率级别统计
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
    
    return sinr_results


def test_semantic_ee_range():
    """测试Semantic-EE范围"""
    print("\n" + "=" * 80)
    print("📊 Semantic-EE范围测试")
    print("=" * 80)
    
    args = get_args()
    n_veh = 6
    n_RB = 10
    
    env = Environ(n_veh=n_veh, n_RB=n_RB, 
                  optimization_target='EE',
                  beta=0.5, circuit_power=0.06)
    
    # 初始化环境
    env.new_random_game()
    env.renew_BS_channel()
    env.renew_BS_channels_fastfading()
    
    # 测试不同功率和压缩比组合
    power_levels_dB = [24, 18, 12, 6, 0]
    rho_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    semantic_ee_results = []
    
    # 测试多个随机位置
    for test_round in range(10):
        env.new_random_game()
        env.renew_BS_channel()
        env.renew_BS_channels_fastfading()
        
        for power_dB in power_levels_dB:
            for rho in rho_levels:
                # 创建测试动作：每个UAV选择不同的RB
                actions_all = np.zeros([n_veh, 3], dtype='float32')
                for i in range(n_veh):
                    actions_all[i, 0] = i % n_RB  # 不同RB避免碰撞
                    actions_all[i, 1] = power_dB  # 功率
                    actions_all[i, 2] = rho  # 压缩比
                
                # 计算性能
                results = env.Compute_Performance_Reward_Train(actions_all, IS_PPO=True)
                (cellular_Rate, cellular_SINR, SE, EE, 
                 semantic_accuracy, semantic_EE, collisions) = results
                
                # 记录Semantic-EE
                for i in range(n_veh):
                    semantic_ee_results.append({
                        'test_round': test_round,
                        'uav': i,
                        'power_dB': power_dB,
                        'rho': rho,
                        'sinr_linear': cellular_SINR[i],
                        'sinr_dB': 10 * np.log10(cellular_SINR[i]) if cellular_SINR[i] > 0 else -np.inf,
                        'semantic_accuracy': semantic_accuracy[i],
                        'semantic_EE': semantic_EE[i],
                        'success': env.success[i]
                    })
    
    # 统计分析
    semantic_ee_all = [r['semantic_EE'] for r in semantic_ee_results if r['semantic_EE'] > 0]
    
    print(f"\n【Semantic-EE统计】")
    print(f"  样本数: {len(semantic_ee_all)}")
    if semantic_ee_all:
        print(f"  最小值: {np.min(semantic_ee_all):.6f}")
        print(f"  最大值: {np.max(semantic_ee_all):.6f}")
        print(f"  平均值: {np.mean(semantic_ee_all):.6f}")
        print(f"  中位数: {np.median(semantic_ee_all):.6f}")
        print(f"  标准差: {np.std(semantic_ee_all):.6f}")
    
    # 按功率级别统计
    print(f"\n【按功率级别统计Semantic-EE】")
    for power_dB in power_levels_dB:
        power_ee = [r['semantic_EE'] for r in semantic_ee_results 
                   if r['power_dB'] == power_dB and r['semantic_EE'] > 0]
        if power_ee:
            print(f"  功率 {power_dB:2d} dB: "
                  f"min={np.min(power_ee):.6f}, "
                  f"max={np.max(power_ee):.6f}, "
                  f"mean={np.mean(power_ee):.6f}")
    
    # 按压缩比统计
    print(f"\n【按压缩比统计Semantic-EE】")
    for rho in rho_levels:
        rho_ee = [r['semantic_EE'] for r in semantic_ee_results 
                 if r['rho'] == rho and r['semantic_EE'] > 0]
        if rho_ee:
            print(f"  压缩比 {rho:.1f}: "
                  f"min={np.min(rho_ee):.6f}, "
                  f"max={np.max(rho_ee):.6f}, "
                  f"mean={np.mean(rho_ee):.6f}")
    
    # 语义准确度统计
    semantic_acc_all = [r['semantic_accuracy'] for r in semantic_ee_results 
                       if r['semantic_accuracy'] > 0]
    print(f"\n【语义准确度统计】")
    print(f"  样本数: {len(semantic_acc_all)}")
    if semantic_acc_all:
        print(f"  最小值: {np.min(semantic_acc_all):.6f}")
        print(f"  最大值: {np.max(semantic_acc_all):.6f}")
        print(f"  平均值: {np.mean(semantic_acc_all):.6f}")
        print(f"  中位数: {np.median(semantic_acc_all):.6f}")
    
    # 成功情况下的Semantic-EE
    success_ee = [r['semantic_EE'] for r in semantic_ee_results 
                 if r['success'] == 1 and r['semantic_EE'] > 0]
    print(f"\n【成功情况下的Semantic-EE】")
    print(f"  样本数: {len(success_ee)}")
    if success_ee:
        print(f"  最小值: {np.min(success_ee):.6f}")
        print(f"  最大值: {np.max(success_ee):.6f}")
        print(f"  平均值: {np.mean(success_ee):.6f}")
        print(f"  中位数: {np.median(success_ee):.6f}")
    
    return semantic_ee_results


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🔬 环境指标测试")
    print("=" * 80)
    print("\n测试配置:")
    print("  - UAV数量: 6")
    print("  - RB数量: 10")
    print("  - 测试轮数: 10 (随机位置)")
    print("  - 功率范围: 0-24 dB")
    print("  - 压缩比范围: 0.1-0.9")
    print("\n")
    
    # 测试SINR范围
    sinr_results = test_sinr_range()
    
    # 测试Semantic-EE范围
    semantic_ee_results = test_semantic_ee_range()
    
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
    
    # Semantic-EE总结
    semantic_ee_all = [r['semantic_EE'] for r in semantic_ee_results if r['semantic_EE'] > 0]
    if semantic_ee_all:
        print(f"\n【Semantic-EE范围】")
        print(f"  范围: {np.min(semantic_ee_all):.6f} ~ {np.max(semantic_ee_all):.6f}")
        print(f"  平均值: {np.mean(semantic_ee_all):.6f}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

