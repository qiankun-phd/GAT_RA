# 回到原始版本的奖励函数

**关键信息**: 用户说"最开始只有信道冲突才有惩罚"

---

## 🔍 原始设计 vs 当前版本

### 原始版本（应该是）
```python
reward = 语义EE + 碰撞惩罚
```
- ✅ 语义EE奖励（主要优化目标）
- ✅ 碰撞惩罚（避免RB冲突）
- ❌ 没有功率惩罚
- ❌ 没有低准确度惩罚

### 当前版本（被修改过）
```python
reward = 语义EE + 碰撞惩罚 + 功率惩罚 + 低准确度惩罚
```

奖励组件占比（当前）:
- 语义EE: 8.75%
- 碰撞惩罚: 35.50%
- **功率惩罚: 6.40%** ← 可能是后加的
- **低准确度惩罚: 49.35%** ← 可能是后加的，且主导了奖励

---

## 🔴 问题根源

### 功率惩罚和低准确度惩罚是后来添加的

这两个惩罚可能是为了：
1. 控制功率消耗（能量效率）
2. 保证语义准确度（通信质量）

**但是**，它们破坏了原始的奖励平衡：
- 低准确度惩罚占49.35%，完全主导了奖励
- 语义EE只占8.75%，几乎被忽略

### 为什么会这样？

**语义EE本身已经包含了这些考虑**：

```
Semantic-EE = Semantic_Accuracy / (P_tx + P_circuit)
```

- 分子：语义准确度（已经在优化通信质量）
- 分母：功率（已经在优化能效）

**所以额外的功率惩罚和低准确度惩罚是重复的！**

---

## ✅ 解决方案：回到原始版本

### 删除后添加的惩罚

只保留：
1. **语义EE奖励**（已包含准确度和能效）
2. **碰撞惩罚**（避免RB冲突）

删除：
1. ❌ 功率惩罚（重复）
2. ❌ 低准确度惩罚（重复）

---

## 📝 代码修改

### 修改 Environment_marl_indoor.py

**简化 `act_for_training` 函数**:

```python
def act_for_training(self, actions, IS_PPO):
    """
    简化版本：只有语义EE和碰撞惩罚
    """
    # ... (前面的代码保持不变)
    
    # Separate reward components for diagnostics
    semantic_accuracy_reward = 0.0
    collision_penalty = 0.0
    
    # Use Semantic Energy Efficiency as reward
    semantic_EE_sum = 0.0
    successful_count = 0
    
    for i in range(len(self.success)):
        if self.success[i] == 1:
            # 成功传输：累加语义EE
            semantic_EE_sum += semantic_EE_penalized[i]
            successful_count += 1
            
            # For diagnostics
            total_power = transmission_power_linear[i] + self.circuit_power
            if total_power > 0:
                semantic_accuracy_reward += semantic_accuracy[i] / total_power
        else:
            # 失败传输：只检查碰撞
            if collisions[i] > 0:
                collision_penalty += self.collision_penalty
    
    # 归一化
    if successful_count > 0:
        reward = semantic_EE_sum / self.n_Veh
        semantic_accuracy_reward = semantic_accuracy_reward / self.n_Veh
    else:
        reward = -1.0 * self.n_Veh  # 全部失败的重惩罚
        semantic_accuracy_reward = 0.0
    
    collision_penalty = collision_penalty / self.n_Veh
    
    # 只返回两个组件
    reward_components = {
        'semantic_accuracy_reward': semantic_accuracy_reward,
        'collision_penalty': collision_penalty
    }
    
    return reward, reward_components
```

### 修改 arguments.py

**删除不需要的参数**:

```python
# 删除或注释掉：
# parser.add_argument('--low_accuracy_penalty', ...)
# parser.add_argument('--accuracy_threshold', ...)

# 只保留：
parser.add_argument('--collision_penalty', type=float, default=-0.5,
                    help='Penalty for RB collision')
```

### 修改 Environment_marl_indoor.py __init__

```python
def __init__(self, n_veh, n_RB, beta=0.5, circuit_power=0.06, 
             optimization_target='SE_EE',
             area_size=1000.0, height_min=50.0, height_max=200.0, 
             comm_range=500.0,
             semantic_A_max=1.0, semantic_beta=2.0, 
             collision_penalty=-0.5):  # 删除 low_accuracy_penalty 和 accuracy_threshold
    
    # ...
    self.collision_penalty = collision_penalty
    # 删除：
    # self.low_accuracy_penalty = low_accuracy_penalty
    # self.accuracy_threshold = accuracy_threshold
```

---

## 📊 预期效果

### 原始版本的奖励占比

删除功率惩罚和低准确度惩罚后：

| 组件 | 预期占比 |
|------|---------|
| 语义EE奖励 | **~70-80%** |
| 碰撞惩罚 | **~20-30%** |

**这才是平衡的奖励函数！**

### 为什么这样更好？

1. **简单**：只有两个组件，易于理解和调试
2. **不重复**：语义EE已包含准确度和能效
3. **平衡**：主要优化目标（语义EE）占主导
4. **符合原始设计**：回到设计者的初衷

---

## 🎯 完整的修改清单

### 删除的内容

1. ❌ 功率惩罚逻辑（line 854-856）
2. ❌ 低准确度惩罚逻辑（line 861-862）
3. ❌ power_penalty变量
4. ❌ low_accuracy_penalty变量
5. ❌ SEMANTIC_REWARD_SCALE放大逻辑
6. ❌ arguments.py中的相关参数

### 保留的内容

1. ✅ 语义EE计算
2. ✅ 碰撞惩罚
3. ✅ 归一化逻辑
4. ✅ TensorBoard诊断（只记录两个组件）

---

## ⚠️ 重要提醒

### 这才是真正的"简化"

回到原始版本：
- 不是"修改"奖励函数
- 而是"恢复"原始设计
- 删除后来添加的复杂逻辑

### 学习率也需要调整

由于奖励范围会改变，建议：
```python
lr_main = 1e-4  # 从 1e-6 提高
```

---

## 📋 实施步骤

1. **备份当前版本**
2. **删除功率惩罚和低准确度惩罚**
3. **更新TensorBoard日志**（只记录两个组件）
4. **调整学习率**
5. **重新训练**
6. **观察前100个episode**

---

**总结**: 

原始设计很可能是正确的！
- 语义EE + 碰撞惩罚
- 简单、清晰、有效

后来添加的功率惩罚和低准确度惩罚：
- 与语义EE重复
- 破坏了奖励平衡
- 导致训练不收敛

**建议：回到原始版本！**

