# 奖励缩放问题诊断报告

**分析日期**: 2025-12-10  
**问题**: 奖励函数缩放严重失衡，导致训练不收敛

---

## 🔴 核心问题

### 奖励组件占比严重失衡

| 组件 | 绝对值均值 | 占比 | 状态 |
|------|-----------|------|------|
| **语义准确度奖励** | 0.0401 | **8.75%** | 🔴 **被严重压制** |
| 功率惩罚 | 0.0293 | 6.40% | ✓ 正常 |
| **碰撞惩罚** | 0.1625 | **35.50%** | ⚠️ 过大 |
| **低准确度惩罚** | 0.2259 | **49.35%** | 🔴 **主导奖励** |

### 关键发现

1. **语义奖励被压制**
   - 语义准确度奖励只占总奖励的8.75%
   - 惩罚项总和是语义奖励的10.42倍
   - **比例**: `1:10.42` (语义奖励:总惩罚)

2. **低准确度惩罚主导**
   - 低准确度惩罚占49.35%，是最大的单一组件
   - 均值-0.226，远大于语义奖励0.040

3. **碰撞惩罚过大**
   - 占35.50%，是第二大组件
   - 均值-0.162，是语义奖励的4倍

---

## 🔍 问题根源

### 当前奖励函数实现

```python
# 环境代码 (Environment_marl_indoor.py)
def act_for_training(self, actions, IS_PPO):
    semantic_accuracy_reward = 0.0
    power_penalty = 0.0
    collision_penalty = 0.0
    low_accuracy_penalty = 0.0
    
    for i in range(len(self.success)):
        if self.success[i] == 1:
            # 语义准确度奖励
            base_semantic_EE = semantic_accuracy[i] / total_power
            semantic_accuracy_reward += base_semantic_EE  # 值很小，约0.04
            
            # 功率惩罚
            power_penalty -= transmission_power_linear[i] * 0.001  # 约-0.03
        else:
            # 失败惩罚
            if collisions[i] > 0:
                collision_penalty += self.collision_penalty  # -0.5 每次
            if semantic_accuracy[i] < self.accuracy_threshold:
                low_accuracy_penalty += self.low_accuracy_penalty  # -0.3 每次
    
    # 归一化
    if successful_count > 0:
        reward = semantic_EE_sum / self.n_Veh  # 除以6
        semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh
    
    # 归一化惩罚
    collision_penalty = collision_penalty / self.n_Veh
    low_accuracy_penalty = low_accuracy_penalty / self.n_Veh
    power_penalty = power_penalty / self.n_Veh
```

### 问题分析

1. **语义EE本身就很小**
   - 语义准确度: 约0.3-0.8
   - 功率: 约0.5-2.0 W
   - `EE = accuracy / power ≈ 0.3-1.5`
   - 归一化后: `EE/6 ≈ 0.05-0.25`

2. **固定惩罚值过大**
   - `collision_penalty = -0.5` (初始化参数)
   - `low_accuracy_penalty = -0.3` (初始化参数)
   - 这些值远大于语义EE

3. **惩罚累积效应**
   - 如果4个UE失败，低准确度惩罚 = -0.3 * 4 / 6 = -0.2
   - 如果这4个UE还有碰撞，碰撞惩罚 = -0.5 * 4 / 6 = -0.33
   - 总惩罚: -0.53，远大于最大语义奖励(约0.1)

---

## 🛠️ 解决方案

### 方案1: 缩放惩罚项（推荐）

**调整惩罚系数，使其与语义奖励同量级**

```python
# 修改 Environment_marl_indoor.py __init__
def __init__(self, ..., 
             collision_penalty=-0.05,      # 从-0.5改为-0.05 (降低10倍)
             low_accuracy_penalty=-0.03,   # 从-0.3改为-0.03 (降低10倍)
             ...)
```

**预期效果**:
- 碰撞惩罚: -0.05 → 占比约3.5%
- 低准确度惩罚: -0.03 → 占比约4.9%
- 语义奖励: 0.04 → 占比约50%+

### 方案2: 放大语义奖励

**在奖励计算中放大语义EE**

```python
# 在 act_for_training 中
if successful_count > 0:
    reward = semantic_EE_sum / self.n_Veh * 10.0  # 放大10倍
    semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh * 10.0
```

**优点**: 不改变惩罚的绝对意义
**缺点**: 可能使奖励过大

### 方案3: 混合方案（最推荐）

**同时调整惩罚和奖励的缩放**

```python
# 调整参数
collision_penalty = -0.1       # 从-0.5改为-0.1 (降低5倍)
low_accuracy_penalty = -0.05   # 从-0.3改为-0.05 (降低6倍)

# 放大语义奖励
semantic_accuracy_reward *= 5.0  # 放大5倍
```

**预期效果**:
- 语义奖励: 0.04 * 5 = 0.2 → 占比约50%
- 碰撞惩罚: -0.1 / 6 ≈ -0.017 → 占比约4%
- 低准确度惩罚: -0.05 / 6 ≈ -0.008 → 占比约2%

### 方案4: 奖励归一化（标准化）

**使用running mean/std进行在线归一化**

```python
class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.gamma = gamma
    
    def normalize(self, reward):
        # 更新统计量
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var = self.gamma * self.var + (1 - self.gamma) * delta ** 2
        
        # 归一化
        std = np.sqrt(self.var + 1e-8)
        return (reward - self.mean) / std

# 在训练循环中
reward_normalizer = RewardNormalizer()
normalized_reward = reward_normalizer.normalize(train_reward)
```

---

## 📊 推荐方案

### 立即实施：方案3（混合方案）

**修改参数**:

1. **降低惩罚系数** (在arguments.py和环境初始化)
   ```python
   parser.add_argument('--collision_penalty', type=float, default=-0.1)    # 从-0.5改为-0.1
   parser.add_argument('--low_accuracy_penalty', type=float, default=-0.05) # 从-0.3改为-0.05
   ```

2. **放大语义奖励** (在Environment_marl_indoor.py的act_for_training)
   ```python
   # 在归一化后
   semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh * 5.0  # 放大5倍
   ```

3. **调整功率惩罚系数**
   ```python
   power_penalty -= transmission_power_linear[i] * 0.0001  # 从0.001改为0.0001 (降低10倍)
   ```

**预期新的占比**:
- 语义准确度奖励: **约50-60%** ✅
- 碰撞惩罚: 约15-20%
- 低准确度惩罚: 约10-15%
- 功率惩罚: 约5-10%

---

## 🎯 实施步骤

### Step 1: 修改arguments.py

```python
parser.add_argument('--collision_penalty', type=float, default=-0.1, 
                    help='Penalty for RB collision (scaled down from -0.5)')
parser.add_argument('--low_accuracy_penalty', type=float, default=-0.05, 
                    help='Penalty for low semantic accuracy (scaled down from -0.3)')
```

### Step 2: 修改Environment_marl_indoor.py

在`act_for_training`函数中:

```python
# 原来:
# semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh

# 修改为:
SEMANTIC_REWARD_SCALE = 5.0  # 放大因子
semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh * SEMANTIC_REWARD_SCALE

# 原来:
# power_penalty -= transmission_power_linear[i] * 0.001

# 修改为:
POWER_PENALTY_SCALE = 0.0001  # 降低功率惩罚
power_penalty -= transmission_power_linear[i] * POWER_PENALTY_SCALE
```

### Step 3: 重新训练并验证

```bash
# 停止当前训练
# 修改代码
# 重新开始训练
python main_PPO_AC.py
```

### Step 4: 监控新的奖励分布

在TensorBoard中查看:
- `Reward/Semantic_Accuracy_Reward` 应该在0.1-0.5范围
- `Reward/Collision_Penalty` 应该在-0.05范围
- `Reward/Low_Accuracy_Penalty` 应该在-0.02范围
- 总奖励应该在-2到+2范围，更容易达到正值

---

## 📝 预期改善

### 训练收敛性

1. **Agent更倾向于探索成功策略**
   - 语义奖励占比提高，成功传输的回报更明显
   - Agent有动力去优化语义准确度

2. **减少过度保守**
   - 惩罚降低，Agent不会过度害怕碰撞
   - 更愿意尝试不同的RB分配

3. **奖励信号更清晰**
   - 各组件比例平衡，学习信号更明确
   - 避免被惩罚项主导

### UE平衡性

- 所有UE的奖励信号更平衡
- 差的UE也能得到足够的学习信号
- 可能改善UE0/UE2/UE5的成功率

---

## ⚠️ 注意事项

1. **需要重新训练**
   - 修改奖励函数后，之前的模型可能不适用
   - 建议从头开始训练

2. **可能需要微调**
   - 如果效果不理想，可以调整放大/缩小因子
   - 建议的范围:
     - `SEMANTIC_REWARD_SCALE`: 3.0-10.0
     - `collision_penalty`: -0.05 to -0.2
     - `low_accuracy_penalty`: -0.03 to -0.1

3. **监控训练**
   - 密切关注前100个episode的奖励分布
   - 确保语义奖励占比在40-60%

---

**报告生成时间**: 2025-12-10  
**优先级**: 🔴 **高** - 这是导致训练不收敛的主要原因

