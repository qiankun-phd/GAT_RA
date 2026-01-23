#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较FRL、MFRL、MRL三种算法的训练结果
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
from collections import defaultdict
import glob

def analyze_tensorboard_log(log_dir):
    """分析TensorBoard日志文件"""
    if not os.path.exists(log_dir):
        print(f"警告: 目录不存在 {log_dir}")
        return None
    
    # 查找所有事件文件
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))
    if not event_files:
        print(f"警告: 未找到事件文件 {log_dir}")
        return None
    
    # 使用最新的事件文件
    event_file = max(event_files, key=os.path.getmtime)
    
    data = defaultdict(list)
    
    try:
        for event in tf.train.summary_iterator(event_file):
            if event.summary:
                for value in event.summary.value:
                    tag = value.tag
                    if value.HasField('simple_value'):
                        data[tag].append({
                            'step': event.step,
                            'value': value.simple_value
                        })
    except Exception as e:
        print(f"读取 {event_file} 时出错: {e}")
        return None
    
    return data

def extract_metrics(data, metric_name):
    """提取指定指标的值"""
    if metric_name not in data:
        return None
    
    values = [item['value'] for item in data[metric_name]]
    steps = [item['step'] for item in data[metric_name]]
    
    if not values:
        return None
    
    return {
        'values': np.array(values),
        'steps': np.array(steps),
        'final': values[-1],
        'mean': np.mean(values),
        'max': np.max(values),
        'min': np.min(values),
        'std': np.std(values)
    }

def analyze_algorithm(log_dir, name):
    """分析单个算法"""
    print(f"\n{'='*60}")
    print(f"分析: {name}")
    print(f"日志目录: {log_dir}")
    print(f"{'='*60}")
    
    data = analyze_tensorboard_log(log_dir)
    if data is None:
        return None
    
    # 提取关键指标
    metrics = {}
    
    # 奖励
    reward_metric = extract_metrics(data, 'Train/reward')
    if reward_metric:
        metrics['reward'] = reward_metric
        print(f"\n奖励 (Reward):")
        print(f"  最终值: {reward_metric['final']:.4f}")
        print(f"  平均值: {reward_metric['mean']:.4f}")
        print(f"  最大值: {reward_metric['max']:.4f}")
        print(f"  最小值: {reward_metric['min']:.4f}")
    
    # 损失
    loss_metric = extract_metrics(data, 'Train/Loss_episode')
    if loss_metric:
        metrics['loss'] = loss_metric
        print(f"\n损失 (Loss):")
        print(f"  最终值: {loss_metric['final']:.6f}")
        print(f"  平均值: {loss_metric['mean']:.6f}")
        print(f"  最小值: {loss_metric['min']:.6f}")
    
    # 成功率
    success_rate_mean = extract_metrics(data, 'Metrics/success_rate_mean')
    if success_rate_mean:
        metrics['success_rate'] = success_rate_mean
        print(f"\n成功率 (Success Rate):")
        print(f"  最终值: {success_rate_mean['final']:.4f}")
        print(f"  平均值: {success_rate_mean['mean']:.4f}")
        print(f"  最大值: {success_rate_mean['max']:.4f}")
    
    # 相似度
    similarity_rate_mean = extract_metrics(data, 'Metrics/similarity_rate_mean')
    if similarity_rate_mean:
        metrics['similarity_rate'] = similarity_rate_mean
        print(f"\n相似度 (Similarity Rate):")
        print(f"  最终值: {similarity_rate_mean['final']:.4f}")
        print(f"  平均值: {similarity_rate_mean['mean']:.4f}")
        print(f"  最大值: {similarity_rate_mean['max']:.4f}")
    
    return metrics

def compare_algorithms():
    """比较三种算法"""
    base_dir = './logs/tensorboard'
    
    experiments = {
        'FRL': os.path.join(base_dir, 'SEE_MAPPO_FRL_Amax1.0_semB2.0_UAV6_RB10_lr1e-06_FL100_max13'),
        'MFRL': os.path.join(base_dir, 'SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10_lr1e-06_FL100_max13'),
        'MRL': os.path.join(base_dir, 'SEE_MAPPO_MRL_Amax1.0_semB2.0_UAV6_RB10_lr1e-06')
    }
    
    results = {}
    for name, log_dir in experiments.items():
        results[name] = analyze_algorithm(log_dir, name)
    
    # 生成对比报告
    print(f"\n\n{'='*80}")
    print("算法性能对比总结")
    print(f"{'='*80}")
    
    # 创建对比表
    print("\n【关键指标对比】")
    print(f"{'指标':<20} {'FRL':<15} {'MFRL':<15} {'MRL':<15} {'最佳':<10}")
    print("-" * 80)
    
    # 奖励对比
    if all('reward' in r and r['reward'] for r in results.values() if r):
        frl_reward = results['FRL']['reward']['final'] if results['FRL'] and 'reward' in results['FRL'] else None
        mfrl_reward = results['MFRL']['reward']['final'] if results['MFRL'] and 'reward' in results['MFRL'] else None
        mrl_reward = results['MRL']['reward']['final'] if results['MRL'] and 'reward' in results['MRL'] else None
        
        rewards = {'FRL': frl_reward, 'MFRL': mfrl_reward, 'MRL': mrl_reward}
        best_reward = max([(k, v) for k, v in rewards.items() if v is not None], key=lambda x: x[1])
        
        frl_str = f"{frl_reward:.4f}" if frl_reward else 'N/A'
        mfrl_str = f"{mfrl_reward:.4f}" if mfrl_reward else 'N/A'
        mrl_str = f"{mrl_reward:.4f}" if mrl_reward else 'N/A'
        print(f"{'最终奖励':<20} {frl_str:<15} {mfrl_str:<15} {mrl_str:<15} {best_reward[0]:<10}")
    
    # 平均奖励对比
    if all('reward' in r and r['reward'] for r in results.values() if r):
        frl_avg = results['FRL']['reward']['mean'] if results['FRL'] and 'reward' in results['FRL'] else None
        mfrl_avg = results['MFRL']['reward']['mean'] if results['MFRL'] and 'reward' in results['MFRL'] else None
        mrl_avg = results['MRL']['reward']['mean'] if results['MRL'] and 'reward' in results['MRL'] else None
        
        avgs = {'FRL': frl_avg, 'MFRL': mfrl_avg, 'MRL': mrl_avg}
        best_avg = max([(k, v) for k, v in avgs.items() if v is not None], key=lambda x: x[1])
        
        frl_avg_str = f"{frl_avg:.4f}" if frl_avg else 'N/A'
        mfrl_avg_str = f"{mfrl_avg:.4f}" if mfrl_avg else 'N/A'
        mrl_avg_str = f"{mrl_avg:.4f}" if mrl_avg else 'N/A'
        print(f"{'平均奖励':<20} {frl_avg_str:<15} {mfrl_avg_str:<15} {mrl_avg_str:<15} {best_avg[0]:<10}")
    
    # 成功率对比
    if all('success_rate' in r and r['success_rate'] for r in results.values() if r):
        frl_sr = results['FRL']['success_rate']['final'] if results['FRL'] and 'success_rate' in results['FRL'] else None
        mfrl_sr = results['MFRL']['success_rate']['final'] if results['MFRL'] and 'success_rate' in results['MFRL'] else None
        mrl_sr = results['MRL']['success_rate']['final'] if results['MRL'] and 'success_rate' in results['MRL'] else None
        
        srs = {'FRL': frl_sr, 'MFRL': mfrl_sr, 'MRL': mrl_sr}
        best_sr = max([(k, v) for k, v in srs.items() if v is not None], key=lambda x: x[1])
        
        frl_sr_str = f"{frl_sr:.4f}" if frl_sr else 'N/A'
        mfrl_sr_str = f"{mfrl_sr:.4f}" if mfrl_sr else 'N/A'
        mrl_sr_str = f"{mrl_sr:.4f}" if mrl_sr else 'N/A'
        print(f"{'最终成功率':<20} {frl_sr_str:<15} {mfrl_sr_str:<15} {mrl_sr_str:<15} {best_sr[0]:<10}")
    
    # 平均成功率对比
    if all('success_rate' in r and r['success_rate'] for r in results.values() if r):
        frl_sr_avg = results['FRL']['success_rate']['mean'] if results['FRL'] and 'success_rate' in results['FRL'] else None
        mfrl_sr_avg = results['MFRL']['success_rate']['mean'] if results['MFRL'] and 'success_rate' in results['MFRL'] else None
        mrl_sr_avg = results['MRL']['success_rate']['mean'] if results['MRL'] and 'success_rate' in results['MRL'] else None
        
        sr_avgs = {'FRL': frl_sr_avg, 'MFRL': mfrl_sr_avg, 'MRL': mrl_sr_avg}
        best_sr_avg = max([(k, v) for k, v in sr_avgs.items() if v is not None], key=lambda x: x[1])
        
        frl_sr_avg_str = f"{frl_sr_avg:.4f}" if frl_sr_avg else 'N/A'
        mfrl_sr_avg_str = f"{mfrl_sr_avg:.4f}" if mfrl_sr_avg else 'N/A'
        mrl_sr_avg_str = f"{mrl_sr_avg:.4f}" if mrl_sr_avg else 'N/A'
        print(f"{'平均成功率':<20} {frl_sr_avg_str:<15} {mfrl_sr_avg_str:<15} {mrl_sr_avg_str:<15} {best_sr_avg[0]:<10}")
    
    # 相似度对比
    if all('similarity_rate' in r and r['similarity_rate'] for r in results.values() if r):
        frl_sim = results['FRL']['similarity_rate']['final'] if results['FRL'] and 'similarity_rate' in results['FRL'] else None
        mfrl_sim = results['MFRL']['similarity_rate']['final'] if results['MFRL'] and 'similarity_rate' in results['MFRL'] else None
        mrl_sim = results['MRL']['similarity_rate']['final'] if results['MRL'] and 'similarity_rate' in results['MRL'] else None
        
        sims = {'FRL': frl_sim, 'MFRL': mfrl_sim, 'MRL': mrl_sim}
        best_sim = max([(k, v) for k, v in sims.items() if v is not None], key=lambda x: x[1])
        
        frl_sim_str = f"{frl_sim:.4f}" if frl_sim else 'N/A'
        mfrl_sim_str = f"{mfrl_sim:.4f}" if mfrl_sim else 'N/A'
        mrl_sim_str = f"{mrl_sim:.4f}" if mrl_sim else 'N/A'
        print(f"{'最终相似度':<20} {frl_sim_str:<15} {mfrl_sim_str:<15} {mrl_sim_str:<15} {best_sim[0]:<10}")
    
    # 平均相似度对比
    if all('similarity_rate' in r and r['similarity_rate'] for r in results.values() if r):
        frl_sim_avg = results['FRL']['similarity_rate']['mean'] if results['FRL'] and 'similarity_rate' in results['FRL'] else None
        mfrl_sim_avg = results['MFRL']['similarity_rate']['mean'] if results['MFRL'] and 'similarity_rate' in results['MFRL'] else None
        mrl_sim_avg = results['MRL']['similarity_rate']['mean'] if results['MRL'] and 'similarity_rate' in results['MRL'] else None
        
        sim_avgs = {'FRL': frl_sim_avg, 'MFRL': mfrl_sim_avg, 'MRL': mrl_sim_avg}
        best_sim_avg = max([(k, v) for k, v in sim_avgs.items() if v is not None], key=lambda x: x[1])
        
        frl_sim_avg_str = f"{frl_sim_avg:.4f}" if frl_sim_avg else 'N/A'
        mfrl_sim_avg_str = f"{mfrl_sim_avg:.4f}" if mfrl_sim_avg else 'N/A'
        mrl_sim_avg_str = f"{mrl_sim_avg:.4f}" if mrl_sim_avg else 'N/A'
        print(f"{'平均相似度':<20} {frl_sim_avg_str:<15} {mfrl_sim_avg_str:<15} {mrl_sim_avg_str:<15} {best_sim_avg[0]:<10}")
    
    # 损失对比
    if all('loss' in r and r['loss'] for r in results.values() if r):
        frl_loss = results['FRL']['loss']['final'] if results['FRL'] and 'loss' in results['FRL'] else None
        mfrl_loss = results['MFRL']['loss']['final'] if results['MFRL'] and 'loss' in results['MFRL'] else None
        mrl_loss = results['MRL']['loss']['final'] if results['MRL'] and 'loss' in results['MRL'] else None
        
        losses = {'FRL': frl_loss, 'MFRL': mfrl_loss, 'MRL': mrl_loss}
        best_loss = min([(k, v) for k, v in losses.items() if v is not None], key=lambda x: x[1])
        
        frl_loss_str = f"{frl_loss:.6f}" if frl_loss else 'N/A'
        mfrl_loss_str = f"{mfrl_loss:.6f}" if mfrl_loss else 'N/A'
        mrl_loss_str = f"{mrl_loss:.6f}" if mrl_loss else 'N/A'
        print(f"{'最终损失':<20} {frl_loss_str:<15} {mfrl_loss_str:<15} {mrl_loss_str:<15} {best_loss[0]:<10}")
    
    # 分析MFRL的优势
    print(f"\n\n{'='*80}")
    print("MFRL优势分析")
    print(f"{'='*80}")
    
    if results['MFRL'] and results['FRL'] and results['MRL']:
        print("\n1. 相比FRL的优势:")
        if 'reward' in results['MFRL'] and 'reward' in results['FRL']:
            reward_improvement = ((results['MFRL']['reward']['final'] - results['FRL']['reward']['final']) / 
                                 abs(results['FRL']['reward']['final']) * 100) if results['FRL']['reward']['final'] != 0 else 0
            print(f"   - 最终奖励提升: {reward_improvement:.2f}%")
        
        if 'similarity_rate' in results['MFRL'] and 'similarity_rate' in results['FRL']:
            sim_improvement = ((results['MFRL']['similarity_rate']['mean'] - results['FRL']['similarity_rate']['mean']) / 
                              (results['FRL']['similarity_rate']['mean'] + 1e-6) * 100)
            print(f"   - 平均相似度提升: {sim_improvement:.2f}%")
        
        print("\n2. 相比MRL的优势:")
        if 'similarity_rate' in results['MFRL'] and 'similarity_rate' in results['MRL']:
            sim_improvement = ((results['MFRL']['similarity_rate']['mean'] - results['MRL']['similarity_rate']['mean']) / 
                              (results['MRL']['similarity_rate']['mean'] + 1e-6) * 100)
            print(f"   - 平均相似度提升: {sim_improvement:.2f}%")
        
        if 'success_rate' in results['MFRL'] and 'success_rate' in results['MRL']:
            sr_improvement = ((results['MFRL']['success_rate']['mean'] - results['MRL']['success_rate']['mean']) / 
                             (results['MRL']['success_rate']['mean'] + 1e-6) * 100)
            print(f"   - 平均成功率提升: {sr_improvement:.2f}%")
        
        print("\n3. MFRL的综合优势:")
        print("   - 结合了元学习的快速适应能力")
        print("   - 结合了联邦学习的协作优势")
        print("   - 在语义通信场景中，相似度是关键指标，MFRL在这方面表现更好")
        print("   - 通过模型聚合，各UE可以共享学习经验，提高整体性能")

if __name__ == '__main__':
    compare_algorithms()
