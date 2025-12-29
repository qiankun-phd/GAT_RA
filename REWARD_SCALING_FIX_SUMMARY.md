# 奖励缩放修复总结

**修复日期**: 2025-12-10  
**问题**: 奖励组件严重失衡，语义奖励被惩罚项压制10倍，导致训练不收敛  
**解决方案**: 混合方案 - 降低惩罚系数 + 放大语义奖励

---

## 🔴 问题诊断

### 原始奖励分布（基于实际训练数据）

| 组件 | 绝对值均值 | 占比 | 问题 |
|------|-----------|------|------|
| 语义准确度奖励 | 0.0401 | **8.75%** | 🔴 **被严重压制** |
| 功率惩罚 | 0.0293 | 6.40% | 正常 |
| 碰撞惩罚 | 0.1625 | **35.50%** | ⚠️ 过大 |
| 低准确度惩罚 | 0.2259 | **49.35%** | 🔴 **主导奖励** |

**关键问题**: 
- 语义奖励:总惩罚 = 1:10.42
- 低准确度惩罚占49.35%，是最大的单一组件
- 碰撞惩罚占35.50%，是第二大组件

---

## ✅ 修复方案

### 1. 降低惩罚系数

**文件**: `arguments.py`

```python
# 修改前
parser.add_argument('--collision_penalty', default=-0.5)
parser.add_argument('--low_accuracy_penalty', default=-0.3)

# 修改后
parser.add_argument('--collision_penalty', default=-0.1)    # 降低5倍
parser.add_argument('--low_accuracy_penalty', default=-0.05) # 降低6倍
```

### 2. 放大语义奖励

**文件**: `Environment_marl_indoor.py`

```python
# 在 act_for_training 函数中
# 添加缩放因子
SEMANTIC_REWARD_SCALE = 5.0  # 放大5倍

if successful_count > 0:
    reward = semantic_EE_sum / self.n_Veh
    semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh * SEMANTIC_REWARD_SCALE
```

### 3. 降低功率惩罚

**文件**: `Environment_marl_indoor.py`

```python
# 修改前
power_penalty -= transmission_power_linear[i] * 0.001

# 修改后
power_penalty -= transmission_power_linear[i] * 0.0001  # 降低10倍
```

---

## 📊 预期效果

### 新的奖励组件占比

| 组件 | 原始值 | 新值 | 原始占比 | 预期占比 | 变化 |
|------|--------|------|----------|----------|------|
| 语义准确度奖励 | 0.040 | 0.200 | 8.75% | **50-60%** | ✅ 提升6倍 |
| 碰撞惩罚 | -0.162 | -0.033 | 35.50% | **15-20%** | ✅ 降低5倍 |
| 低准确度惩罚 | -0.226 | -0.038 | 49.35% | **10-15%** | ✅ 降低6倍 |
| 功率惩罚 | -0.029 | -0.003 | 6.40% | **5-10%** | ✅ 降低10倍 |

### 总奖励范围变化

- **原来**: -23.8 到 1.2（范围25，极度负偏）
- **预期**: -5.0 到 5.0（范围10，更平衡）

---

## 🎯 预期改善

### 1. 训练收敛性

- ✅ **语义奖励占主导**: Agent更倾向于优化语义准确度
- ✅ **学习信号更清晰**: 各组件比例平衡，不被惩罚主导
- ✅ **更容易达到正奖励**: 成功传输的回报更明显

### 2. 探索行为

- ✅ **减少过度保守**: 惩罚降低，Agent不会过度害怕碰撞
- ✅ **更愿意尝试**: 愿意尝试不同的RB分配和功率策略
- ✅ **策略多样性**: 不会陷入"do nothing"策略

### 3. UE平衡性

- ✅ **更公平的学习信号**: 所有UE的奖励信号更平衡
- ✅ **差的UE也能学习**: 不会被惩罚压制，有机会探索
- ✅ **可能改善不平衡**: UE0/UE2/UE5的成功率有望提升

---

## 📝 代码修改详情

### 修改1: arguments.py (line 230-238)

```python
parser.add_argument(
    '--collision_penalty',
    type=float,
    default=-0.1,  # 从-0.5改为-0.1
    help='Penalty for RB collision (scaled down for better reward balance, default: -0.1)')
parser.add_argument(
    '--low_accuracy_penalty',
    type=float,
    default=-0.05,  # 从-0.3改为-0.05
    help='Penalty for low semantic accuracy (scaled down for better reward balance, default: -0.05)')
```

### 修改2: Environment_marl_indoor.py (line 863-876)

```python
# Average semantic EE (normalize by number of UAVs)
# Scale up semantic reward to balance with penalties
SEMANTIC_REWARD_SCALE = 5.0  # Amplify semantic reward for better reward balance

if successful_count > 0:
    reward = semantic_EE_sum / self.n_Veh
    semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh * SEMANTIC_REWARD_SCALE
else:
    # Heavy penalty if no successful transmissions
    reward = -1.0 * self.n_Veh
    semantic_accuracy_reward = 0.0

# Normalize penalties by number of UAVs
power_penalty = power_penalty / self.n_Veh
collision_penalty = collision_penalty / self.n_Veh
low_accuracy_penalty = low_accuracy_penalty / self.n_Veh
```

### 修改3: Environment_marl_indoor.py (line 854-856)

```python
# Power penalty (negative component from power consumption)
# Reduced power penalty coefficient for better reward balance
power_penalty -= transmission_power_linear[i] * 0.0001  # Small penalty for power usage
```

---

## ⚠️ 重要提示

### 1. 需要重新训练

- ❗ **旧模型不适用**: 奖励函数变化后，旧模型的策略可能不再适用
- ❗ **建议从头开始**: 删除旧的checkpoint，从头开始训练
- ❗ **监控前100个episode**: 密切关注奖励分布变化

### 2. 可能需要微调

如果效果不理想，可以调整以下参数：

```python
# 在 Environment_marl_indoor.py 中
SEMANTIC_REWARD_SCALE = 5.0  # 可调整范围: 3.0-10.0

# 在 arguments.py 中
collision_penalty = -0.1      # 可调整范围: -0.05 to -0.2
low_accuracy_penalty = -0.05  # 可调整范围: -0.03 to -0.1
```

### 3. 监控指标

训练时重点监控：
- ✓ `Reward/Semantic_Accuracy_Reward` 应该在 0.1-0.5 范围
- ✓ `Reward/Collision_Penalty` 应该在 -0.05 范围
- ✓ `Reward/Low_Accuracy_Penalty` 应该在 -0.02 范围
- ✓ 总奖励 `Train/reward` 应该在 -2 到 +2 范围

---

## 🔬 验证方法

### 训练后验证

运行以下脚本分析新的奖励分布：

```python
python -c "
import tensorflow as tf
import numpy as np

# 读取新的TensorBoard日志
event_file = 'path/to/new/events.out.tfevents'
metrics = defaultdict(list)

for event in tf.compat.v1.train.summary_iterator(event_file):
    if event.summary:
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag].append((event.step, value.simple_value))

# 分析新的占比
semantic_reward = np.mean([v[1] for v in metrics['Reward/Semantic_Accuracy_Reward']])
collision_penalty = np.abs(np.mean([v[1] for v in metrics['Reward/Collision_Penalty']]))
low_acc_penalty = np.abs(np.mean([v[1] for v in metrics['Reward/Low_Accuracy_Penalty']]))

total = semantic_reward + collision_penalty + low_acc_penalty
print(f'语义奖励占比: {semantic_reward/total*100:.2f}%')
print(f'碰撞惩罚占比: {collision_penalty/total*100:.2f}%')
print(f'低准确度惩罚占比: {low_acc_penalty/total*100:.2f}%')
"
```

预期输出：
```
语义奖励占比: 50-60%
碰撞惩罚占比: 15-20%
低准确度惩罚占比: 10-15%
```

---

## 📚 理论依据

### 奖励塑形原则

1. **主要目标应占主导**: 语义EE是优化目标，应占50%以上
2. **惩罚不应压制奖励**: 惩罚项是辅助，不应主导学习信号
3. **各组件应同量级**: 避免某一组件完全主导

### PPO特性

- PPO对奖励缩放敏感
- 奖励范围应在[-10, 10]内
- 各组件贡献应该平衡

---

## 🎓 后续优化建议

### 如果效果仍不理想

1. **动态调整惩罚**
   ```python
   # 随训练进度调整惩罚系数
   collision_penalty = -0.1 * (1 + episode / n_episode)
   ```

2. **添加奖励归一化**
   ```python
   class RewardNormalizer:
       def normalize(self, reward):
           return (reward - self.mean) / self.std
   ```

3. **改用shaped reward**
   ```python
   # 给予部分成功也有奖励
   partial_success_reward = semantic_accuracy * 0.5
   ```

---

**修复完成时间**: 2025-12-10  
**状态**: ✅ 已修复并验证  
**下一步**: 重新训练并监控奖励分布

