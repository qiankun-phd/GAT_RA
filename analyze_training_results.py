#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练结果：比较FRL、MFRL、MRL三种方法的性能
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
from collections import defaultdict

def analyze_experiment(log_dir, name):
    """分析单个实验的训练结果"""
    # 查找事件文件
    event_file = None
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(log_dir, file)
                break
    
    if not event_file or not os.path.exists(event_file):
        return None
    
    data = defaultdict(list)
    
    try:
        for event in tf.train.summary_iterator(event_file):
            if event.summary:
                for value in event.summary.value:
                    tag = value.tag
                    if value.HasField('simple_value'):
                        data[tag].append({
                            'step': event.step,
                            'wall_time': event.wall_time,
                            'value': value.simple_value
                        })
    except Exception as e:
        print(f"读取 {event_file} 时出错: {e}")
        return None
    
    return data

def main():
    base_dir = '/home/qiankun/Semcom_ra/GAT_RA/logs/tensorboard'
    
    experiments = {
        'FRL (联邦学习)': 'SEE_MAPPO_FRL_Amax1.0_semB2.0_UAV6_RB10',
        'MFRL (元学习+联邦学习)': 'SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10',
        'MRL (元学习)': 'SEE_MAPPO_MRL_Amax1.0_semB2.0_UAV6_RB10'
    }
    
    all_results = {}
    
    # 读取所有实验结果
    print("="*80)
    print("读取训练日志...")
    print("="*80)
    
    for name, exp_name in experiments.items():
        log_dir = os.path.join(base_dir, exp_name)
        print(f"\n读取: {name}")
        data = analyze_experiment(log_dir, name)
        if data:
            all_results[name] = data
            print(f"  成功读取 {len(data)} 个指标，共 {sum(len(v) for v in data.values())} 个数据点")
        else:
            print(f"  读取失败")
    
    if not all_results:
        print("没有成功读取任何实验结果")
        return
    
    # 详细分析每个实验
    print("\n" + "="*80)
    print("详细分析")
    print("="*80)
    
    for name, data in all_results.items():
        print(f"\n【{name}】")
        print("-"*80)
        
        # 按重要性排序指标
        priority_tags = [tag for tag in sorted(data.keys()) 
                        if any(kw in tag.lower() for kw in ['reward', 'loss', 'entropy', 'value', 'success', 'similarity'])]
        
        for tag in priority_tags:
            values = [d['value'] for d in data[tag]]
            steps = [d['step'] for d in data[tag]]
            
            if not values:
                continue
            
            print(f"\n  {tag}:")
            print(f"    步数范围: {steps[0]} - {steps[-1]} (共 {len(steps)} 个数据点)")
            print(f"    初始值: {values[0]:.6f}")
            print(f"    最终值: {values[-1]:.6f}")
            print(f"    平均值: {np.mean(values):.6f}")
            print(f"    最大值: {np.max(values):.6f}")
            print(f"    最小值: {np.min(values):.6f}")
            print(f"    标准差: {np.std(values):.6f}")
            
            # 计算趋势
            if len(values) > 10:
                n_10 = max(1, len(values) // 10)
                early_avg = np.mean(values[:n_10])
                late_avg = np.mean(values[-n_10:])
                improvement = ((late_avg - early_avg) / abs(early_avg) * 100) if early_avg != 0 else 0
                print(f"    改进幅度: {improvement:+.2f}% (最后10% vs 前10%)")
    
    # 比较分析
    print("\n" + "="*80)
    print("性能比较总结")
    print("="*80)
    
    # 查找所有指标
    all_tags = set()
    for data in all_results.values():
        all_tags.update(data.keys())
    
    # 重点比较的指标
    key_tags = [
        'Train/reward',
        'Train/Loss_episode',
        'Metrics/success_rate_mean',
        'Metrics/similarity_rate_mean'
    ]
    
    # 添加所有Metrics相关的指标
    for tag in sorted(all_tags):
        if tag.startswith('Metrics/') and tag not in key_tags:
            key_tags.append(tag)
    
    for tag in key_tags:
        if tag not in all_tags:
            continue
        
        print(f"\n【{tag}】")
        print("-"*80)
        print(f"{'方法':<30} {'最终值':<15} {'平均值':<15} {'最大值':<15} {'改进%':<12}")
        print("-"*80)
        
        results_list = []
        for name in ['FRL (联邦学习)', 'MFRL (元学习+联邦学习)', 'MRL (元学习)']:
            if name in all_results and tag in all_results[name]:
                data = all_results[name][tag]
                values = [d['value'] for d in data]
                steps = [d['step'] for d in data]
                
                final = values[-1]
                avg = np.mean(values)
                max_val = np.max(values)
                min_val = np.min(values)
                
                # 计算改进幅度
                if len(values) > 10:
                    n_10 = max(1, len(values) // 10)
                    early_avg = np.mean(values[:n_10])
                    improvement = ((avg - early_avg) / abs(early_avg) * 100) if early_avg != 0 else 0
                else:
                    improvement = 0
                
                results_list.append({
                    'name': name,
                    'final': final,
                    'avg': avg,
                    'max': max_val,
                    'min': min_val,
                    'improvement': improvement
                })
                
                print(f"{name:<30} {final:<15.6f} {avg:<15.6f} {max_val:<15.6f} {improvement:>+10.2f}%")
        
        # 找出最佳方法
        if results_list:
            best_final = max(results_list, key=lambda x: x['final'])
            best_avg = max(results_list, key=lambda x: x['avg'])
            print(f"\n  最佳最终值: {best_final['name']} ({best_final['final']:.6f})")
            print(f"  最佳平均值: {best_avg['name']} ({best_avg['avg']:.6f})")
    
    # 总结
    print("\n" + "="*80)
    print("总结与建议")
    print("="*80)
    
    # 比较关键指标
    if 'Train/reward' in all_tags:
        print("\n【奖励 (Reward) 分析】")
        reward_results = {}
        for name in all_results.keys():
            if 'Train/reward' in all_results[name]:
                values = [d['value'] for d in all_results[name]['Train/reward']]
                reward_results[name] = {
                    'final': values[-1],
                    'avg': np.mean(values),
                    'max': np.max(values)
                }
        
        if reward_results:
            best = max(reward_results.items(), key=lambda x: x[1]['final'])
            print(f"  最佳最终奖励: {best[0]} ({best[1]['final']:.6f})")
            best_avg = max(reward_results.items(), key=lambda x: x[1]['avg'])
            print(f"  最佳平均奖励: {best_avg[0]} ({best_avg[1]['avg']:.6f})")
    
    if 'Metrics/success_rate_mean' in all_tags:
        print("\n【成功率 (Success Rate) 分析】")
        success_results = {}
        for name in all_results.keys():
            if 'Metrics/success_rate_mean' in all_results[name]:
                values = [d['value'] for d in all_results[name]['Metrics/success_rate_mean']]
                success_results[name] = {
                    'final': values[-1],
                    'avg': np.mean(values),
                    'max': np.max(values)
                }
        
        if success_results:
            best = max(success_results.items(), key=lambda x: x[1]['final'])
            print(f"  最佳最终成功率: {best[0]} ({best[1]['final']:.6f})")
            best_avg = max(success_results.items(), key=lambda x: x[1]['avg'])
            print(f"  最佳平均成功率: {best_avg[0]} ({best_avg[1]['avg']:.6f})")

if __name__ == '__main__':
    main()
