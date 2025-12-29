# 最小化语义通信适配总结

**日期**: 2025-12-10  
**原则**: 在原始代码结构上做最小改动

---

## 🎯 适配策略

### 原始代码结构（保持不变）
```python
# 原始 act_for_training
for i in range(len(self.success)):
    if (self.success[i] == 1) and (cellular_SINR[i] > training_sinr_threshold):
        SE_sum += SE[i]
        EE_sum += EE[i]
    else:
        # 失败惩罚
        SE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
        EE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
        break

reward = (self.beta * SE_sum + (1 - self.beta) * EE_sum) / self.n_Veh
```

### 适配后（最小修改）
```python
# 适配语义通信
for i in range(len(self.success)):
    if (self.success[i] == 1) and (cellular_SINR[i] > training_sinr_threshold):
        SE_sum += SE[i]
        Semantic_EE_sum += semantic_EE[i]  # 用Semantic-EE替换EE
    else:
        # 失败惩罚（保持不变）
        SE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
        Semantic_EE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
        break

reward = (self.beta * SE_sum + (1 - self.beta) * Semantic_EE_sum) / self.n_Veh
```

---

## ✅ 修改内容

### 1. Environment_marl_indoor.py

#### 删除的参数（__init__）
```python
# 删除
collision_penalty=-0.5
low_accuracy_penalty=-0.3  
accuracy_threshold=0.5
```

#### 简化的 act_for_training
```python
def act_for_training(self, actions, IS_PPO):
    """最小化适配：只用Semantic-EE替换EE"""
    # 计算性能指标
    results = self.Compute_Performance_Reward_Train(action_temp, is_ppo_mode)
    (cellular_Rate, cellular_SINR, SE, EE, 
     semantic_accuracy, semantic_EE, collisions) = results
    
    # 与原始代码相同的结构
    SE_sum = 0.0
    Semantic_EE_sum = 0.0
    training_sinr_threshold = 3.3
    
    for i in range(len(self.success)):
        if (self.success[i] == 1) and (cellular_SINR[i] > training_sinr_threshold):
            SE_sum += SE[i]
            Semantic_EE_sum += semantic_EE[i]  # 唯一的关键修改
        else:
            SE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
            Semantic_EE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
            break
    
    # 与原始代码相同的奖励计算
    if self.optimization_target == 'SE':
        reward = SE_sum / self.n_Veh
    elif self.optimization_target == 'EE':
        reward = Semantic_EE_sum / self.n_Veh
    elif self.optimization_target == 'SE_EE':
        reward = (self.beta * SE_sum + (1 - self.beta) * Semantic_EE_sum) / self.n_Veh
    else:
        reward = (self.beta * SE_sum + (1 - self.beta) * Semantic_EE_sum) / self.n_Veh
    
    return reward
```

#### 删除的penalties逻辑
```python
# 删除
penalties = np.zeros(self.n_Veh)
for i in range(len(self.vehicles)):
    if collisions[i] > 0:
        penalties[i] += self.collision_penalty
    if semantic_accuracy[i] < self.accuracy_threshold:
        penalties[i] += self.low_accuracy_penalty

# 改为
semantic_EE_penalized = semantic_EE  # 不添加额外惩罚
```

### 2. arguments.py

#### 删除的参数
```python
# 删除（或注释）
# parser.add_argument('--collision_penalty', ...)
# parser.add_argument('--low_accuracy_penalty', ...)
# parser.add_argument('--accuracy_threshold', ...)
```

### 3. main_PPO_AC.py

#### 环境初始化（需要更新）
```python
# 原来
env = Environ(n_veh, n_RB, ..., 
              collision_penalty=args.collision_penalty,
              low_accuracy_penalty=args.low_accuracy_penalty,
              accuracy_threshold=args.accuracy_threshold)

# 改为
env = Environ(n_veh, n_RB, ...,
              semantic_A_max=args.semantic_A_max,
              semantic_beta=args.semantic_beta)
```

---

## 🔄 对比：复杂版本 vs 简单版本

### 复杂版本（之前）
- ❌ SEMANTIC_REWARD_SCALE放大
- ❌ 功率惩罚
- ❌ 低准确度惩罚
- ❌ 碰撞惩罚单独处理
- ❌ reward_components字典
- ❌ 复杂的诊断逻辑

### 简单版本（现在）
- ✅ 只用Semantic-EE替换EE
- ✅ 保持原始奖励结构
- ✅ 失败惩罚通过success标志处理
- ✅ 简单清晰，易于理解

---

## 📊 为什么这样更好？

### 1. Semantic-EE已包含所有考虑

```
Semantic-EE = Semantic_Accuracy / (P_tx + P_circuit)
               ↑优化准确度        ↑优化能效
```

- **分子**：语义准确度（通信质量）
- **分母**：功率消耗（能量效率）
- 不需要额外的惩罚项

### 2. 原始失败机制已经足够

```python
if (self.success[i] == 1) and (cellular_SINR[i] > threshold):
    # 成功：正奖励
else:
    # 失败：负惩罚
    break
```

- 碰撞 → success[i] = 0 → 失败
- SINR不足 → 失败
- 不需要额外的collision_penalty

### 3. 简单就是美

- 代码少 → bug少
- 逻辑清晰 → 易于理解
- 参数少 → 易于调优
- 符合原始设计理念

---

## ⚠️ 需要更新的文件

### 必须修改
1. ✅ Environment_marl_indoor.py
   - __init__: 删除3个参数
   - act_for_training: 简化为原始结构
   - Compute_Performance_Reward_Train: 删除penalties

2. ⏳ main_PPO_AC.py
   - 环境初始化：删除传递的参数

3. ⏳ arguments.py  
   - 删除或注释相关参数

### 可选修改
- simulate()函数：可能不再需要返回reward_components
- TensorBoard日志：简化为只记录总奖励

---

## 🎯 预期效果

### 奖励组件
```
reward = beta * SE + (1-beta) * Semantic-EE
```

- 当 beta=0.5:
  - SE占50%
  - Semantic-EE占50%
  - 简单平衡

### 失败处理
```
success=0 或 SINR<3.3 → 负惩罚
```
- 与原始代码完全一致
- 简单有效

---

## 📝 TODO

- [ ] 更新main_PPO_AC.py中的环境初始化
- [ ] 删除arguments.py中的相关参数
- [ ] 简化simulate()函数（可选）
- [ ] 简化TensorBoard日志（可选）
- [ ] 重新训练测试

---

**总结**: 
回到原始简洁的设计，只做最小的语义通信适配。
用Semantic-EE替换EE，保持其他逻辑不变。

