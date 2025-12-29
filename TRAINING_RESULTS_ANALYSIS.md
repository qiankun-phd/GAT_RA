# 训练结果分析报告

**训练模式**: EE_MAPPO_FRL (能效优化 + 联邦学习)  
**训练集数**: 517 Episodes  
**网络模式**: MLP (action_dim=3: RB + Power + Rho)  
**分析时间**: 2025-12-10

---

## 📊 整体训练表现

### 训练进度
- **Episodes**: 517 (已完成)
- **Loss**: 0.0158 (稳定收敛) ✅
- **Reward**: -0.7983 (从-0.9867提升19.09%) ↑
- **平均成功率**: 20.17% (从1.33%提升1412%) ⚠️

### 趋势分析
```
Loss:   0.0365 → 0.0158 (-56.68%) ✅ 收敛良好
Reward: -0.987 → -0.798 (+19.09%) ↑ 持续改进
成功率: 1.33% → 20.17% (+1412%)  ⚠️ 仍然较低
```

---

## 🎯 关键指标详情

### 1. 损失函数 (Loss)
- **初始**: 0.0365
- **最终**: 0.0158
- **最低**: 0.0157 (Episode 495)
- **标准差**: 0.000233 (最近20个episode)

**分析**: ✅ Loss已稳定收敛，训练过程稳定

### 2. 奖励 (Reward)
- **初始**: -0.9867
- **最终**: -0.7983
- **最高**: -0.6967 (Episode 453)
- **前10平均**: -0.9788
- **后10平均**: -0.7857

**分析**: ↑ 奖励持续改进，但仍为负值，说明整体性能有待提升

### 3. 成功率 (Success Rate)
- **初始**: 1.33%
- **最终**: 20.17%
- **最高**: 30.33% (Episode 453)
- **平均**: 14.16%

**分析**: ⚠️ 成功率较低，只有约1/5的传输成功

---

## 👥 各UE性能分析

### UE性能对比表

| UE | 当前成功率 | 平均成功率 | 最高成功率 | 状态 |
|----|-----------|-----------|-----------|------|
| **UE 0** | 0.00% | 0.01% | 2.00% | ❌ 几乎失败 |
| **UE 1** | 0.00% | 0.26% | 6.00% | ❌ 几乎失败 |
| **UE 2** | 0.00% | 0.01% | 1.00% | ❌ 几乎失败 |
| **UE 3** | **95.00%** | **66.98%** | **100.00%** | ✅ 表现优秀 |
| **UE 4** | 0.00% | 0.01% | 1.00% | ❌ 几乎失败 |
| **UE 5** | 26.00% | 17.68% | 85.00% | ⚠️ 中等 |

### 性能分布
```
优秀 (>50%):   1个UE (UE 3)           - 16.7%
中等 (10-50%): 1个UE (UE 5)           - 16.7%
较差 (<10%):   4个UE (UE 0,1,2,4)     - 66.7%
```

**严重问题**: 🚨 **成功率极度不平衡！**

---

## 🔍 问题诊断

### 🚨 主要问题

#### 1. 成功率不平衡（核心问题）
**现象**:
- UE 3: 95% 成功率（主导者）
- UE 5: 26% 成功率（次要）
- UE 0,1,2,4: 0% 成功率（失败）

**可能原因**:
- **信道质量差异**: UE 3可能位置最优，信道条件最好
- **资源分配不公**: UE 3可能抢占了大部分资源
- **学习偏差**: 联邦学习中UE 3的策略主导了全局模型
- **RB冲突**: 弱势UE总是与强势UE冲突，无法成功传输

#### 2. 整体成功率低 (20%)
**现象**: 即使训练了517个episodes，成功率仍只有20%

**可能原因**:
- 环境难度太高（SINR阈值过高？）
- 奖励信号不够明确
- 学习率过低（1e-6）导致学习缓慢
- 语义通信的rho参数尚未学会合理取值

#### 3. Reward仍为负值
**现象**: Reward从-0.987改进到-0.798，但仍为负

**说明**: 
- 系统整体性能仍需大幅提升
- 可能失败惩罚过重
- Semantic-EE值偏低

---

## 💡 改进建议

### 优先级1: 解决成功率不平衡 🔥

#### 方案A: 调整联邦学习权重
```python
# 在 PPO_brain_AC.py 的 averaging_model 中
# 降低高成功率UE的权重，增加低成功率UE的学习机会
weights = np.ones(self.n_veh) / self.n_veh  # 当前：均等权重

# 改为：
weights = 1.0 / (success_rate + 0.1)  # 反比例权重
weights = weights / weights.sum()
```

#### 方案B: 独立训练阶段
```python
# 前N个episodes不使用联邦学习，让各UE独立学习
if i_episode < 200:  # 独立训练前200个episodes
    # 不调用 averaging_model
    pass
else:
    ppoes.averaging_model(success_rate)
```

#### 方案C: 公平性约束
```python
# 在环境中添加公平性奖励
fairness_bonus = -np.std(success_rate_all_UEs)  # 惩罚不平衡
reward += fairness_bonus
```

### 优先级2: 提升学习效率 ⚡

#### 增加学习率
```python
# arguments.py
parser.add_argument('--lr_main', type=float, default=5e-6)  # 从1e-6提升到5e-6
```

#### 增加Entropy权重
```python
# arguments.py  
parser.add_argument('--weight_for_entropy', type=float, default=0.02)  # 从0.01提升到0.02
```

### 优先级3: 优化奖励函数 🎯

#### 检查SINR阈值
```python
# Environment_marl_indoor.py
training_sinr_threshold = 3.3  # 当前阈值
# 如果太高，降低到2.5或3.0
```

#### 平衡奖励分量
```python
# 检查 Semantic-EE 的计算
# 确保压缩比 rho 的影响合理
```

### 优先级4: 诊断Rho学习 🔬

#### 添加Rho监控
```python
# main_PPO_AC.py 中添加
summary.value.add(tag='Actions/rho_mean', simple_value=float(np.mean(actions[:, 2])))
summary.value.add(tag='Actions/rho_std', simple_value=float(np.std(actions[:, 2])))
```

#### 检查Rho分布
```python
# 打印rho值，查看是否学到有意义的压缩策略
print(f"Rho values: {action_all_training[:, 2]}")
```

---

## 🔬 深入诊断步骤

### Step 1: 检查信道质量
```python
# 在 simulate() 中添加
cellular_SINR = env.cellular_SINR
print(f"SINR per UE: {cellular_SINR}")
```

### Step 2: 检查RB分配
```python
# 打印各UE选择的RB
print(f"RB selection: {action_all_training[:, 0]}")
# 检查是否有严重冲突
```

### Step 3: 检查Rho取值
```python
# 打印各UE的压缩比
print(f"Rho values: {action_all_training[:, 2]}")
# 预期：应该在[0.3, 0.9]之间动态调整
```

### Step 4: 检查语义准确度
```python
# 在 Environment 中打印
print(f"Semantic Accuracy: {semantic_accuracy}")
# 检查与rho的关系
```

---

## 📈 训练策略建议

### 短期（立即实施）

1. **降低学习率衰减**: 保持探索性
2. **增加Entropy权重**: 避免过早收敛
3. **监控Rho值**: 确认是否学到语义压缩策略

### 中期（调整后观察）

1. **调整联邦学习策略**: 使用加权平均或延迟启动
2. **优化奖励函数**: 增加公平性约束
3. **调整SINR阈值**: 如果环境过难

### 长期（根据效果迭代）

1. **启用GAT模式**: 测试图注意力网络是否能改善协调
2. **多阶段训练**: 先独立训练，再联邦学习
3. **动态难度**: 根据成功率动态调整环境参数

---

## 🎯 下一步行动

### 立即执行：

1. **添加诊断日志**:
```python
# 在 main_PPO_AC.py 的 simulate() 中
print(f"Episode {i_episode}: RB={action_all_training[:,0]}, Rho={action_all_training[:,2]}")
print(f"  SINR: {env.cellular_SINR}")
print(f"  Success: {env.success}")
```

2. **调整学习率**:
```bash
python main_PPO_AC.py --lr_main 5e-6 --weight_for_entropy 0.02
```

3. **禁用早期联邦学习**:
```python
# main_PPO_AC.py
if IS_FL and i_episode % target_average_step == 0 and i_episode >= 200:  # 延迟到200 episodes后
    ppoes.averaging_model(success_rate)
```

### 实验对比：

| 实验 | 配置 | 目标 |
|-----|------|------|
| **Baseline** | 当前配置 | 基准性能 |
| **Exp 1** | lr=5e-6 | 加快学习 |
| **Exp 2** | 延迟FL (200 eps) | 改善平衡 |
| **Exp 3** | SINR=3.0 | 降低难度 |
| **Exp 4** | 公平性奖励 | 强制平衡 |

---

## 📝 总结

### ✅ 优点
- Loss收敛稳定
- Reward持续改进
- UE 3学习效果优秀（95%成功率）
- 训练过程无崩溃

### ⚠️ 问题
- **严重的成功率不平衡**（核心问题）
- 整体成功率偏低（20%）
- 4个UE几乎完全失败
- Reward仍为负值

### 🎯 核心目标
**解决成功率不平衡，让所有UE都能有效学习！**

### 🚀 建议路径
```
1. 添加诊断 → 定位问题根源
2. 调整联邦学习 → 改善UE平衡
3. 提升学习率 → 加快收敛
4. 监控Rho → 验证语义通信
5. 迭代优化 → 持续改进
```

---

*分析完成时间: 2025-12-10*  
*基于: 517 Episodes训练数据*

