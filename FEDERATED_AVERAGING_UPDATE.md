# 联邦学习聚合方式更新

**更新日期**: 2025-12-10  
**更新内容**: 将语义加权平均改为标准平均加权（Uniform Averaging）

---

## 📋 更新内容

### 修改前：语义加权平均

```python
# 根据语义EE计算权重
weights = normalize(semantic_EE_weights)
θ_global = Σ_i w_i * θ_i
```

**问题**:
- 性能好的agent权重更大，可能主导全局模型
- 性能差的agent权重很小，可能被忽略
- 可能导致UE不平衡问题加剧

### 修改后：标准平均加权（FedAvg）

```python
# 所有agent权重相等
weights = [1/n_veh, 1/n_veh, ..., 1/n_veh]
θ_global = (1/n_veh) * Σ_i θ_i
```

**优势**:
- ✅ 简单稳定
- ✅ 公平对待所有agent
- ✅ 避免性能好的agent主导全局模型
- ✅ 符合标准联邦学习协议（FedAvg）
- ✅ 可能有助于解决UE不平衡问题

---

## 🔧 代码修改

### 1. PPO_brain_AC.py

**修改前**:
```python
def averaging_model(self, success_rate, semantic_EE_weights=None):
    if semantic_EE_weights is not None:
        # 使用语义EE加权
        weights = normalize(semantic_EE_weights)
    else:
        # 回退到平均加权
        weights = np.ones(self.n_veh) / self.n_veh
```

**修改后**:
```python
def averaging_model(self, success_rate, semantic_EE_weights=None):
    # 始终使用平均加权（标准FedAvg）
    weights = np.ones(self.n_veh) / self.n_veh
    print(f"Federated Averaging: using uniform weights (1/{self.n_veh} for each agent)")
    
    # 语义EE仅用于日志记录（可选）
    if semantic_EE_weights is not None:
        print(f"Semantic EE per agent (for reference): {semantic_EE_weights}")
```

### 2. main_PPO_AC.py

**修改前**:
```python
# Use semantic-EE weighted averaging
ppoes.averaging_model(success_rate, semantic_EE_weights=avg_semantic_EE_all)
```

**修改后**:
```python
# Use uniform averaging (equal weights for all agents)
ppoes.averaging_model(success_rate, semantic_EE_weights=None)
```

---

## 📊 聚合公式

### 标准Federated Averaging (FedAvg)

```
θ_global = (1/n) * Σ_{i=1}^n θ_i
```

其中：
- `n = n_veh`: agent数量（默认6）
- `θ_i`: 第i个agent的模型参数
- `w_i = 1/n`: 所有agent权重相等

### 对于GAT网络

聚合所有GAT层的参数：
- GAT编码器参数
- Actor网络参数（Power, RB, Compression）
- Critic网络参数

### 对于MLP网络

聚合所有MLP层的参数：
- 隐藏层权重和偏置
- Actor输出层（Power, RB, Compression）
- Critic输出层

---

## 🎯 预期效果

### 1. 公平性提升

- **之前**: UE3权重大，其他UE权重小
- **现在**: 所有UE权重相等，公平聚合

### 2. 可能解决UE不平衡

- **之前**: 语义加权可能加剧不平衡（好的更好，差的更差）
- **现在**: 平均加权可能帮助差的UE学习到好的策略

### 3. 训练稳定性

- **之前**: 权重计算可能不稳定（语义EE为负时）
- **现在**: 权重固定，更稳定

---

## 📝 使用说明

### 当前配置

- **聚合方式**: 标准平均加权（FedAvg）
- **聚合频率**: 每 `target_average_step` 个episode
- **聚合时机**: 训练的前90%阶段

### 日志输出

当执行模型聚合时，会输出：
```
Federated Averaging: using uniform weights (1/6 for each agent)
Semantic EE per agent (for reference): [-0.43, -0.45, -0.43, -0.00, -0.43, -0.47]
```

### 如需切换回语义加权

如果需要切换回语义加权，可以：
1. 修改 `averaging_model` 函数，恢复语义加权逻辑
2. 在 `main_PPO_AC.py` 中传递 `semantic_EE_weights=avg_semantic_EE_all`

---

## 🔍 验证

### 检查聚合是否正确

在训练日志中查找：
```
Federated Averaging: using uniform weights (1/6 for each agent)
```

### 检查权重

所有agent的权重应该都是 `1/6 ≈ 0.1667`

---

## 📚 参考

- **FedAvg论文**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- **标准协议**: 所有agent权重相等，简单平均

---

**更新完成时间**: 2025-12-10  
**状态**: ✅ 已更新并验证

