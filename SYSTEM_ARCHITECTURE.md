# 系统架构与算法详解

## 📋 目录

1. [系统概述](#系统概述)
2. [算法架构](#算法架构)
3. [环境详情](#环境详情)
4. [代码逻辑流程](#代码逻辑流程)
5. [关键组件说明](#关键组件说明)
6. [训练流程](#训练流程)
7. [联邦学习机制](#联邦学习机制)

---

## 系统概述

### 问题定义

**多智能体无人机网络资源分配与语义通信优化**

- **场景**: 室内环境中的多无人机（UAV）网络
- **任务**: 每个UAV需要选择资源块（RB）、传输功率和语义压缩比
- **目标**: 最大化语义能量效率（Semantic Energy Efficiency, Semantic-EE）
- **挑战**: 
  - 多智能体协作与竞争
  - 动态信道条件
  - 资源冲突避免
  - 语义通信质量保证

### 系统组成

```
┌─────────────────────────────────────────────────────────┐
│                   训练主循环 (main_PPO_AC.py)            │
│  - 环境交互                                              │
│  - 数据收集                                              │
│  - 模型训练                                              │
│  - 联邦学习聚合                                          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              PPO智能体 (PPO_brain_AC.py)                 │
│  - GAT编码器 (Graph Attention Network)                  │
│  - Actor网络 (策略网络)                                  │
│  - Critic网络 (值函数网络)                               │
│  - PPO损失计算                                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           环境 (Environment_marl_indoor.py)              │
│  - UAV位置与移动                                         │
│  - 信道模型 (A2G)                                        │
│  - 语义通信计算                                          │
│  - 奖励计算                                              │
└─────────────────────────────────────────────────────────┘
```

---

## 算法架构

### 1. 多智能体近端策略优化 (MAPPO)

#### 核心思想

- **PPO (Proximal Policy Optimization)**: 策略梯度算法，通过裁剪机制防止策略更新过大
- **Multi-Agent**: 每个UAV是一个独立的智能体，共享环境但独立决策
- **Centralized Training, Decentralized Execution**: 训练时可以使用全局信息，执行时只使用局部信息

#### PPO损失函数

```
L = L_clip_power + L_clip_RB + L_clip_rho - c1 * L_vf + c2 * S
```

其中：
- `L_clip_power`: 功率动作的PPO裁剪损失
- `L_clip_RB`: RB选择的PPO裁剪损失
- `L_clip_rho`: 压缩比的PPO裁剪损失
- `L_vf`: 值函数损失（Critic）
- `S`: 策略熵（鼓励探索）
- `c1`: 值函数损失权重
- `c2`: 熵权重

#### 动作空间

每个UAV的动作是3维向量：
- **RB选择** (Resource Block): 离散动作，范围 [0, n_RB-1]
- **传输功率** (Power): 连续动作，范围 [-RB_action_bound, RB_action_bound] (dB)
- **压缩比** (Compression Ratio, ρ): 连续动作，范围 [0, 1]

#### 状态表示

**GAT模式** (默认):
- **节点特征**: 每个UAV是一个节点
  - CSI快衰落: `[n_RB]` - 每个RB的信道状态
  - CSI慢衰落: `[n_RB]` - 路径损耗和阴影
  - 位置信息: `[3]` - (x, y, z)坐标
  - 成功标志: `[1]` - 上次传输是否成功
  - Episode进度: `[1]` - 当前episode进度
  - **总维度**: `n_RB * 2 + 3 + 2 = 35` (当n_RB=10时)

- **邻接矩阵**: `[n_veh, n_veh]` - 基于通信距离的图拓扑
  - 如果两个UAV距离 < `comm_range` (默认500m)，则连接

**MLP模式** (向后兼容):
- 扁平化的状态向量，包含所有UAV的信息

### 2. 图注意力网络 (GAT)

#### 架构

```
输入: [batch, n_veh, node_feature_dim]
  ↓
GAT编码器 (多层)
  ├─ Layer 1: [n_veh, hidden_1] × num_heads
  ├─ Layer 2: [n_veh, hidden_2] × num_heads
  └─ Layer 3: [n_veh, hidden_3] × num_heads
  ↓
节点嵌入: [n_veh, hidden_3 * num_heads]
  ↓
  ├─ Actor输入: [batch * n_veh, hidden_dim]
  │   └─ 每个节点独立输出动作
  │
  └─ Critic输入: [batch, hidden_dim]
      └─ 图级聚合 (mean pooling)
```

#### 注意力机制

对于节点 `i` 和 `j`，注意力权重计算：

```
e_ij = LeakyReLU(W_a^T [W_h h_i || W_h h_j])
α_ij = softmax_j(e_ij)
h_i' = σ(Σ_j α_ij W_h h_j)
```

其中：
- `h_i`: 节点i的特征
- `W_h`: 特征变换矩阵
- `W_a`: 注意力权重矩阵
- `||`: 拼接操作
- `α_ij`: 归一化的注意力权重

#### 多头注意力

使用多个注意力头（默认4个），每个头学习不同的关系：
```
h_i' = ||_k=1^K σ(Σ_j α_ij^k W_h^k h_j)
```

最终输出是多个头的拼接。

### 3. 语义通信

#### 语义准确度 (Semantic Accuracy)

基于压缩比 `ρ` 和信噪比 `SINR` 计算：

```
A(ρ, SINR) = A_max * (1 - exp(-β * ρ)) * log(1 + SINR) / log(1 + max_SINR)
```

其中：
- `A_max`: 最大语义准确度（默认1.0）
- `β`: 压缩比敏感度参数（默认2.0）
- `ρ`: 压缩比 [0, 1]
- `SINR`: 信噪比（线性尺度）
- `max_SINR`: 最大期望SINR（默认100，约20dB）

**物理意义**:
- **压缩比项** `(1 - exp(-β * ρ))`: 
  - 压缩比越高（ρ→1），语义信息保留越多，准确度越高
  - β控制敏感度，β越大，压缩比的影响越大
- **SINR项** `log(1 + SINR) / log(1 + max_SINR)`:
  - SINR越高，信道质量越好，准确度越高
  - 使用对数尺度平滑SINR的影响
- **乘积形式**: 两者相乘，需要同时满足高压缩比和高SINR才能获得高准确度

#### 语义能量效率 (Semantic-EE)

```
Semantic-EE = A(ρ, SINR) / (P_tx + P_circuit)
```

其中：
- `P_tx`: 传输功率（线性尺度）
- `P_circuit`: 电路功耗（默认0.06）

**优化目标**: 最大化语义准确度，同时最小化能耗

#### 惩罚机制

1. **碰撞惩罚** (`collision_penalty = -0.5`):
   - 当多个UAV选择相同的RB时触发
   - 鼓励RB分配多样性

2. **低准确度惩罚** (`low_accuracy_penalty = -0.3`):
   - 当语义准确度 < `accuracy_threshold` (默认0.5) 时触发
   - 保证最低通信质量

3. **失败传输惩罚**:
   - 当所有UAV传输失败时，给予重惩罚 `-n_veh`
   - 鼓励至少部分UAV成功传输

---

## 环境详情

### 环境类型: 室内环境 (Indoor)

#### 空间设置

- **区域大小**: 1000m × 1000m × (50-200)m (3D空间)
- **UAV数量**: 6个（可配置）
- **资源块数**: 10个（可配置）
- **通信范围**: 500m（用于构建图拓扑）

#### UAV模型

每个UAV具有：
- **位置**: (x, y, z) 3D坐标
- **速度**: (vx, vy, vz) 3D速度向量
- **移动模式**: 随机游走（可扩展为其他模式）

#### 信道模型: Air-to-Ground (A2G)

##### 路径损耗

**LoS (视距)**:
```
PL_LoS = 28.0 + 22*log10(d) + 20*log10(fc)
```

**NLoS (非视距)**:
```
PL_NLoS = -17.5 + (46-7*log10(h_BS))*log10(d) + 20*log10(40*π*fc/3)
```

其中：
- `d`: 3D距离 (m)
- `fc`: 载波频率 (GHz)
- `h_BS`: 基站高度 (m)

##### LoS概率

```
P_LoS = 1 / (1 + a * exp(-b * (θ - a)))
```

其中：
- `θ`: 仰角 (度)
- `a, b`: 环境相关参数

##### 快衰落

- **LoS**: Rician分布（K因子取决于距离）
- **NLoS**: Rayleigh分布

##### 阴影衰落

- 对数正态分布
- 标准差: 8.29 dB

#### 资源块 (RB) 配置

```
RB索引: 0  1  2  3  4  5  6  7  8  9
带宽:   0.18 0.18 0.36 0.36 0.36 0.72 0.72 0.72 1.44 1.44 (MHz)
```

#### 功率等级

```
功率等级: [24, 21, 18, 15, 12, 9, 6, 3, 0] (dBm)
```

#### 成功传输条件

1. **无碰撞**: 没有其他UAV选择相同的RB
2. **SINR阈值**: `SINR > 3.16 dB` (线性尺度: ~2.07)

#### 时间设置

- **快衰落更新**: 每1ms (`time_fast = 0.001s`)
- **慢衰落/位置更新**: 每100ms (`time_slow = 0.1s`)
- **每个Episode**: `T_TIMESTEPS = time_slow / time_fast = 100` 步

---

## 代码逻辑流程

### 主训练循环 (`main_PPO_AC.py`)

```
1. 初始化
   ├─ 环境创建
   ├─ PPO智能体创建
   └─ TensorBoard日志设置

2. Episode循环 (n_episode次)
   ├─ 并行执行多个simulate() (ACTOR_NUM个)
   │   └─ 收集经验数据
   │
   ├─ 数据聚合
   │   ├─ 奖励
   │   ├─ 成功率
   │   └─ 语义EE
   │
   ├─ 模型训练
   │   ├─ 采样批次数据
   │   ├─ 计算GAE
   │   └─ PPO更新 (K次)
   │
   ├─ TensorBoard日志
   │   ├─ 损失组件
   │   ├─ 策略熵
   │   ├─ 梯度范数
   │   ├─ 奖励组件
   │   └─ 动作分布
   │
   └─ 联邦学习聚合 (如果启用)
       └─ 语义加权平均
```

### 模拟函数 (`simulate()`)

```
for step in range(T_TIMESTEPS):
    1. 更新环境
       ├─ 更新UAV位置
       ├─ 更新信道快衰落
       └─ 更新图拓扑
    
    2. 获取图数据
       ├─ 节点特征提取
       └─ 邻接矩阵构建
    
    3. 动作选择
       ├─ 每个UAV使用GAT网络选择动作
       └─ 动作: [RB, Power, Compression Ratio]
    
    4. 环境执行
       ├─ 计算SINR
       ├─ 检测碰撞
       ├─ 计算语义准确度
       └─ 计算奖励
    
    5. 数据存储
       ├─ 状态
       ├─ 动作
       ├─ 奖励
       └─ 值函数预测

返回: (平均奖励, 经验数据, 成功率, 语义EE, 奖励组件)
```

### PPO训练 (`PPO_brain_AC.py`)

```
1. 网络构建 (_build_net)
   ├─ GAT编码器
   ├─ Actor网络
   │   ├─ Power分布 (Normal)
   │   ├─ RB分布 (Categorical)
   │   └─ Compression分布 (Beta)
   └─ Critic网络

2. 动作选择 (choose_action)
   ├─ 从分布中采样
   └─ 返回: [RB_index, power, compression_ratio]

3. 值函数估计 (get_v)
   └─ 返回状态值

4. 训练 (train)
   ├─ 计算概率比率
   ├─ 计算GAE优势
   ├─ 计算PPO损失
   ├─ 计算梯度
   └─ 更新参数 (K次)
```

---

## 关键组件说明

### 1. 图数据构建 (`get_graph_data()`)

```python
def get_graph_data(env, n_veh, ind_episode=0.):
    """
    构建图结构数据
    返回:
        node_features: [n_veh, node_feature_dim]
        adj_matrix: [n_veh, n_veh]
    """
```

**节点特征**:
- CSI快衰落: `(channels_fast - channels_abs + 10) / 35` (归一化)
- CSI慢衰落: `(channels_abs - 80) / 60.0` (归一化)
- 位置: `[x/width, y/height, z/depth]` (归一化)
- 成功标志: `[success]`
- Episode进度: `[ind_episode / n_episode]`

**邻接矩阵**:
- 基于UAV间距离
- 如果 `distance < comm_range`，则 `adj[i,j] = 1`，否则 `0`

### 2. GAT编码器 (`multi_layer_gat()`)

```python
def multi_layer_gat(node_features, adj_matrix, hidden_dims, num_heads=4):
    """
    多层图注意力网络
    输入:
        node_features: [n_veh, node_feature_dim]
        adj_matrix: [n_veh, n_veh]
    输出:
        node_embeddings: [n_veh, hidden_dim * num_heads]
    """
```

**特点**:
- 支持多层堆叠
- 多头注意力机制
- 残差连接（可选）
- Dropout正则化（可选）

### 3. 语义准确度计算 (`compute_semantic_accuracy()`)

```python
def compute_semantic_accuracy(self, rho, sinr):
    """
    计算语义准确度 (mAP)
    参数:
        rho: 压缩比 [0, 1]
        sinr: 信噪比 (线性尺度)
    返回:
        accuracy: 语义准确度 [0, A_max]
    """
    accuracy = self.semantic_A_max * (1 - np.exp(-self.semantic_beta * rho * sinr))
    return accuracy
```

### 4. 奖励计算 (`act_for_training()`)

```python
def act_for_training(self, actions, IS_PPO):
    """
    执行动作并计算奖励
    返回:
        reward: 平均语义EE（带惩罚）
        reward_components: 分离的奖励组件
    """
```

**奖励组成**:
1. **语义准确度奖励**: 成功传输的语义EE总和
2. **功率惩罚**: 功率消耗的负贡献
3. **碰撞惩罚**: RB冲突的惩罚
4. **低准确度惩罚**: 准确度低于阈值的惩罚

### 5. 联邦学习聚合 (`averaging_model()`)

```python
def averaging_model(self, success_rate, semantic_EE_weights=None):
    """
    基于语义EE的加权模型平均
    权重计算:
        w_i = (EE_i - min(EE)) / sum(EE_j - min(EE))
    """
```

**聚合频率**: 每 `target_average_step` 个episode执行一次

**聚合方式**:
- **GAT模式**: 聚合所有GAT层参数
- **MLP模式**: 聚合所有MLP层参数

---

## 训练流程

### 超参数配置

**PPO参数**:
- `epsilon = 0.2`: PPO裁剪范围
- `c1 = 0.5`: 值函数损失权重
- `c2 = 0.01`: 熵权重
- `lr = 1e-4`: 学习率
- `K = 10`: 每次更新的epoch数
- `gamma = 0.99`: 折扣因子
- `lambda_advantage = 0.95`: GAE参数

**GAT参数**:
- `num_gat_heads = 4`: 注意力头数
- `hidden_dims = [128, 64, 32]`: 隐藏层维度

**环境参数**:
- `n_veh = 6`: UAV数量
- `n_RB = 10`: 资源块数
- `T_TIMESTEPS = 100`: 每个episode的步数

### 训练模式

1. **标准RL** (`IS_FL=False, IS_meta=False`):
   - 独立训练每个智能体
   - 无模型聚合

2. **联邦强化学习 (FRL)** (`IS_FL=True, IS_meta=False`):
   - 定期聚合模型参数
   - 使用语义EE加权平均

3. **元强化学习 (MRL)** (`IS_FL=False, IS_meta=True`):
   - 元学习快速适应新任务
   - 需要多任务环境

4. **元联邦强化学习 (MFRL)** (`IS_FL=True, IS_meta=True`):
   - 结合元学习和联邦学习

### 诊断指标

**TensorBoard日志**:
- `Loss/Actor_Loss`: Actor损失
- `Loss/Critic_Loss`: Critic损失
- `Policy/Entropy`: 策略熵
- `Training/Gradient_Norm`: 梯度范数
- `Reward/Semantic_Accuracy_Reward`: 语义准确度奖励
- `Reward/Power_Penalty`: 功率惩罚
- `Reward/Collision_Penalty`: 碰撞惩罚
- `Action/Rho_Mean`: 压缩比均值
- `Action/Rho_Std`: 压缩比标准差

---

## 联邦学习机制

### 语义加权平均

**权重计算**:
```python
# 1. 获取每个agent的平均语义EE
semantic_EE_weights = [EE_0, EE_1, ..., EE_n-1]

# 2. 归一化权重
weights = (semantic_EE_weights - min(semantic_EE_weights)) / sum(...)

# 3. 加权聚合
θ_global = Σ_i w_i * θ_i
```

**优势**:
- 性能好的agent权重更大
- 鼓励所有agent提升性能
- 避免性能差的agent拖累全局模型

### 聚合时机

- **频率**: 每 `target_average_step` 个episode
- **条件**: 在训练的前90%阶段执行
- **目的**: 避免后期过度聚合导致性能下降

---

## 文件结构

```
GAT_RA/
├── main_PPO_AC.py              # 主训练脚本
├── PPO_brain_AC.py             # PPO智能体实现
├── Environment_marl_indoor.py   # 室内环境实现
├── arguments.py                # 参数配置
├── meta_train_PPO_AC.py        # 元学习训练脚本
├── meta_brain_PPO.py           # 元学习PPO实现
├── analyze_tensorboard.py      # TensorBoard分析工具
├── logs/                       # 训练日志
│   └── tensorboard/           # TensorBoard事件文件
├── model/                      # 保存的模型
└── Train_data/                 # 训练数据
```

---

## 关键设计决策

### 1. 为什么使用GAT？

- **图结构**: UAV网络天然形成图结构
- **关系建模**: GAT可以学习UAV间的协作/竞争关系
- **可扩展性**: 支持动态图拓扑变化
- **信息共享**: 通过注意力机制共享邻居信息

### 2. 为什么使用语义通信？

- **效率**: 语义通信比传统通信更高效
- **质量**: 关注语义准确度而非比特准确度
- **应用**: 适合AI任务（如图像识别、目标检测）

### 3. 为什么使用MAPPO？

- **稳定性**: PPO的裁剪机制保证训练稳定
- **多智能体**: 支持多智能体协作
- **效率**: 样本效率高

### 4. 为什么使用联邦学习？

- **隐私**: 不需要共享原始数据
- **协作**: 多个agent可以协作学习
- **鲁棒性**: 对单个agent的失败更鲁棒

---

## 总结

本系统实现了一个基于GAT和MAPPO的多智能体语义通信资源分配框架，具有以下特点：

1. **图结构建模**: 使用GAT建模UAV网络拓扑
2. **语义通信**: 优化语义能量效率而非传统通信指标
3. **多智能体协作**: MAPPO支持多智能体训练
4. **联邦学习**: 语义加权平均提升全局性能
5. **完整诊断**: TensorBoard提供详细的训练诊断

---

**文档版本**: 1.0  
**最后更新**: 2025-12-10

