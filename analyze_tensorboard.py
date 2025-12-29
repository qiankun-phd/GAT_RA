#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorBoard事件文件分析工具
分析训练过程中的各项指标
"""

import tensorflow as tf
import numpy as np
import sys1
import os
from collections import defaultdict

def analyze_tensorboard_event(event_file_path):
    """分析TensorBoard事件文件"""
    
    if not os.path.exists(event_file_path):
        print(f"错误: 文件不存在: {event_file_path}")
        return
    
    # 存储所有指标
    metrics = defaultdict(list)
    
    try:
        print(f"正在读取: {event_file_path}")
        for event in tf.compat.v1.train.summary_iterator(event_file_path):
            if event.summary:
                for value in event.summary.value:
                    tag = value.tag
                    step = event.step
                    if value.HasField('simple_value'):
                        metrics[tag].append((step, value.simple_value))
        
        if not metrics:
            print("警告: 未找到任何指标数据")
            return
        
        print("\n" + "="*60)
        print("TensorBoard训练结果分析")
        print("="*60 + "\n")
        
        # 统计信息
        print("【数据统计】")
        print("-" * 60)
        for tag in sorted(metrics.keys()):
            values = [v[1] for v in metrics[tag]]
            if values:
                steps = [v[0] for v in metrics[tag]]
                print(f"\n{tag}:")
                print(f"  数据点数: {len(values)}")
                print(f"  Episode范围: {min(steps)} - {max(steps)}")
                print(f"  最新值 (Episode {max(steps)}): {values[-1]:.6f}")
                print(f"  平均值: {np.mean(values):.6f}")
                print(f"  最大值: {np.max(values):.6f} (Episode {steps[np.argmax(values)]})")
                print(f"  最小值: {np.min(values):.6f} (Episode {steps[np.argmin(values)]})")
                if len(values) > 1:
                    change = values[-1] - values[0]
                    change_pct = (change / abs(values[0]) * 100) if values[0] != 0 else 0
                    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(f"  趋势: {trend} {change:.6f} ({change_pct:.2f}%)")
        
        # 训练指标详细分析
        print("\n" + "="*60)
        print("【训练指标详细分析】")
        print("="*60)
        
        if 'Train/reward' in metrics:
            rewards = [v[1] for v in metrics['Train/reward']]
            steps = [v[0] for v in metrics['Train/reward']]
            print(f"\n奖励 (Reward):")
            print(f"  初始值 (Episode {steps[0]}): {rewards[0]:.4f}")
            print(f"  最终值 (Episode {steps[-1]}): {rewards[-1]:.4f}")
            change = rewards[-1] - rewards[0]
            change_pct = (change / abs(rewards[0]) * 100) if rewards[0] != 0 else 0
            print(f"  总提升: {change:.4f} ({change_pct:.2f}%)")
            print(f"  最高值: {np.max(rewards):.4f} (Episode {steps[np.argmax(rewards)]})")
            print(f"  最低值: {np.min(rewards):.4f} (Episode {steps[np.argmin(rewards)]})")
            
            # 计算最近N个episode的平均值
            if len(rewards) >= 10:
                recent_avg = np.mean(rewards[-10:])
                early_avg = np.mean(rewards[:10])
                print(f"  前10个episode平均: {early_avg:.4f}")
                print(f"  后10个episode平均: {recent_avg:.4f}")
                print(f"  改进: {recent_avg - early_avg:.4f}")
        
        if 'Train/Loss_episode' in metrics:
            losses = [v[1] for v in metrics['Train/Loss_episode']]
            steps = [v[0] for v in metrics['Train/Loss_episode']]
            print(f"\n损失 (Loss):")
            print(f"  初始值 (Episode {steps[0]}): {losses[0]:.6f}")
            print(f"  最终值 (Episode {steps[-1]}): {losses[-1]:.6f}")
            change = losses[-1] - losses[0]
            print(f"  变化: {change:.6f}")
            print(f"  最低值: {np.min(losses):.6f} (Episode {steps[np.argmin(losses)]})")
            print(f"  最高值: {np.max(losses):.6f} (Episode {steps[np.argmax(losses)]})")
            
            # 检查损失是否稳定
            if len(losses) >= 20:
                recent_std = np.std(losses[-20:])
                print(f"  最近20个episode标准差: {recent_std:.6f}")
                if recent_std < 0.01:
                    print("  ⚠ 损失值变化很小，可能已收敛或需要调整学习率")
        
        # 成功率分析
        print("\n" + "="*60)
        print("【成功率分析】")
        print("="*60)
        
        if 'Metrics/success_rate_mean' in metrics:
            success_rates = [v[1] for v in metrics['Metrics/success_rate_mean']]
            steps = [v[0] for v in metrics['Metrics/success_rate_mean']]
            print(f"\n平均成功率:")
            print(f"  初始值 (Episode {steps[0]}): {success_rates[0]:.4f}")
            print(f"  最终值 (Episode {steps[-1]}): {success_rates[-1]:.4f}")
            change = success_rates[-1] - success_rates[0]
            change_pct = (change / abs(success_rates[0]) * 100) if success_rates[0] != 0 else 0
            print(f"  提升: {change:.4f} ({change_pct:.2f}%)")
            print(f"  最高值: {np.max(success_rates):.4f} (Episode {steps[np.argmax(success_rates)]})")
            
            # 各UE成功率
            print(f"\n各UE成功率 (最新值):")
            for i in range(6):
                tag = f'Metrics/success_rate_ue_{i}'
                if tag in metrics:
                    values = [v[1] for v in metrics[tag]]
                    print(f"  UE {i}: {values[-1]:.4f} (平均: {np.mean(values):.4f}, 最高: {np.max(values):.4f})")
        
        # 语义EE分析
        print("\n" + "="*60)
        print("【语义能量效率 (Semantic-EE) 分析】")
        print("="*60)
        
        if 'Semantic/semantic_EE_mean' in metrics:
            semantic_EE = [v[1] for v in metrics['Semantic/semantic_EE_mean']]
            steps = [v[0] for v in metrics['Semantic/semantic_EE_mean']]
            print(f"\n平均语义EE:")
            print(f"  初始值 (Episode {steps[0]}): {semantic_EE[0]:.6f}")
            print(f"  最终值 (Episode {steps[-1]}): {semantic_EE[-1]:.6f}")
            change = semantic_EE[-1] - semantic_EE[0]
            change_pct = (change / abs(semantic_EE[0]) * 100) if semantic_EE[0] != 0 else 0
            print(f"  提升: {change:.6f} ({change_pct:.2f}%)")
            print(f"  最高值: {np.max(semantic_EE):.6f} (Episode {steps[np.argmax(semantic_EE)]})")
            print(f"  最低值: {np.min(semantic_EE):.6f} (Episode {steps[np.argmin(semantic_EE)]})")
            
            # 计算最近N个episode的平均值
            if len(semantic_EE) >= 10:
                recent_avg = np.mean(semantic_EE[-10:])
                early_avg = np.mean(semantic_EE[:10])
                print(f"  前10个episode平均: {early_avg:.6f}")
                print(f"  后10个episode平均: {recent_avg:.6f}")
                print(f"  改进: {recent_avg - early_avg:.6f}")
        
        if 'Semantic/semantic_EE_max' in metrics:
            semantic_EE_max = [v[1] for v in metrics['Semantic/semantic_EE_max']]
            print(f"\n最大语义EE: {np.max(semantic_EE_max):.6f}")
        
        if 'Semantic/semantic_EE_min' in metrics:
            semantic_EE_min = [v[1] for v in metrics['Semantic/semantic_EE_min']]
            print(f"最小语义EE: {np.min(semantic_EE_min):.6f}")
        
        # 各UE的语义EE
        print(f"\n各UE语义EE (最新值):")
        ue_ee_values = []
        for i in range(6):
            tag = f'Semantic/semantic_EE_ue_{i}'
            if tag in metrics:
                values = [v[1] for v in metrics[tag]]
                latest = values[-1]
                ue_ee_values.append(latest)
                print(f"  UE {i}: {latest:.6f} (平均: {np.mean(values):.6f}, 最高: {np.max(values):.6f})")
        
        if ue_ee_values:
            print(f"\n各UE语义EE统计:")
            print(f"  平均值: {np.mean(ue_ee_values):.6f}")
            print(f"  标准差: {np.std(ue_ee_values):.6f}")
            print(f"  最大值: {np.max(ue_ee_values):.6f} (UE {np.argmax(ue_ee_values)})")
            print(f"  最小值: {np.min(ue_ee_values):.6f} (UE {np.argmin(ue_ee_values)})")
        
        # 训练建议
        print("\n" + "="*60)
        print("【训练建议】")
        print("="*60)
        
        suggestions = []
        
        if 'Train/reward' in metrics:
            rewards = [v[1] for v in metrics['Train/reward']]
            if len(rewards) >= 20:
                recent_avg = np.mean(rewards[-10:])
                early_avg = np.mean(rewards[:10])
                if recent_avg <= early_avg * 1.05:  # 提升小于5%
                    suggestions.append("⚠ 奖励提升缓慢，考虑调整学习率或探索策略")
        
        if 'Train/Loss_episode' in metrics:
            losses = [v[1] for v in metrics['Train/Loss_episode']]
            if len(losses) >= 20:
                recent_std = np.std(losses[-20:])
                if recent_std < 0.01:
                    suggestions.append("⚠ 损失值已稳定，可能已收敛")
        
        if 'Metrics/success_rate_mean' in metrics:
            success_rates = [v[1] for v in metrics['Metrics/success_rate_mean']]
            if success_rates[-1] < 0.5:
                suggestions.append("⚠ 成功率较低，检查环境设置和奖励函数")
        
        if 'Semantic/semantic_EE_mean' in metrics:
            semantic_EE = [v[1] for v in metrics['Semantic/semantic_EE_mean']]
            if len(semantic_EE) >= 20:
                recent_avg = np.mean(semantic_EE[-10:])
                early_avg = np.mean(semantic_EE[:10])
                if recent_avg <= early_avg * 1.05:
                    suggestions.append("⚠ 语义EE提升缓慢，考虑优化资源分配策略")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        else:
            print("✓ 训练指标正常，继续观察")
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        event_file = sys.argv[1]
    else:
        # 默认使用最新的TensorBoard事件文件
        event_file = './logs/tensorboard/GAT_heads4_Semantic_EE_MAPPO_FRL_A1.0_beta2.0_UAV6_RB10/events.out.tfevents.1765267417.network-ra'
    
    analyze_tensorboard_event(event_file)


