# 最小化修改方案

**基于**: `/home/qiankun/GAT_RA/origin/` 原始代码  
**目标**: 添加语义通信（压缩比rho）和GAT开关  
**原则**: 保持原始训练流程，做最小改动

---

## 📋 原始代码结构

### main_PPO_AC.py
```python
# 动作维度
action_dim = 2  # RB_choice + power

# 训练数据
action_all_training = np.zeros([n_veh, 2])  # [RB, power]

# 执行动作
train_reward = env.act_for_training(action_temp, IS_PPO)

# 训练
loss = ppoes.train(s, a, gae, reward, v_pred_next, sess)
```

### PPO_brain_AC.py
```python
# 网络输出
mu, sigma = ...  # Power (Normal分布)
RB_probs = ...   # RB (Categorical分布)

# 动作采样
choose_action_op = concat([RB_sample, power_sample])  # [RB, power]
```

---

## ✅ 最小化修改方案

### 1. 添加压缩比输出（PPO_brain_AC.py）

#### 网络定义
```python
# 在 _build_net 中添加
self.w_rho_alpha = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
self.w_rho_beta = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
self.b_rho_alpha = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
self.b_rho_beta = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)

# 添加Beta分布（用于压缩比 rho ∈ [0,1]）
rho_alpha = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_alpha), self.b_rho_alpha)) + 1.0
rho_beta = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_beta), self.b_rho_beta)) + 1.0
rho_distribution = tf.distributions.Beta(rho_alpha, rho_beta)

# 返回
return norm_dist, RB_distribution, rho_distribution, v, params, saver
```

#### 动作采样
```python
# 修改 choose_action_op
self.choose_action_op = tf.concat([
    tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)),
    tf.squeeze(pi.sample(1), axis=0),
    tf.squeeze(rho_distribution.sample(1), axis=0)  # 添加rho
], 1)
```

#### Loss函数
```python
# 在原有基础上添加rho的loss
rho_action = self.a[:,2]
ratio_rho = rho_distribution.prob(rho_action) / old_rho_distribution.prob(rho_action)
L_rho = tf.reduce_mean(tf.minimum(
    ratio_rho * GAE_advantage,
    tf.clip_by_value(ratio_rho, 1 - epsilon, 1 + epsilon) * GAE_advantage
))

# 更新总Loss
L = L_clip + L_RB + L_rho - c1 * L_vf + c2 * S
```

### 2. 添加GAT开关

#### PPO_brain_AC.py
```python
def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, meta_lr, K, n_veh, n_RB, 
             IS_meta, meta_episode, use_gat=False):
    self.use_gat = use_gat
    # ...
    
    if use_gat:
        # GAT模式
        pi, RB_dist, rho_dist, self.v, params, self.saver = self._build_net_gat(...)
    else:
        # MLP模式（原始）
        pi, RB_dist, rho_dist, self.v, params, self.saver = self._build_net(...)
```

### 3. 更新main_PPO_AC.py

```python
# 动作维度
action_dim = 3  # RB_choice + power + rho

# 动作边界
action_bound = [n_RB, args.RB_action_bound, 1.0]  # 添加rho边界

# 训练数据
action_all_training = np.zeros([n_veh, 3])  # [RB, power, rho]

# PPO初始化
use_gat = args.use_gat if hasattr(args, 'use_gat') else False
ppoes = PPO(state_dim, action_bound, ..., use_gat=use_gat)
```

---

## 🔧 具体修改步骤

### Step 1: 修改PPO_brain_AC.py

1. 添加rho相关的网络参数（6行）
2. 修改`_build_net`返回值（1行）
3. 修改`choose_action_op`（1行）
4. 修改Loss函数（5行）
5. 修改`choose_action`的裁剪（1行）
6. 修改`averaging_model`（2行）

**总共约15行代码修改**

### Step 2: 修改main_PPO_AC.py

1. 修改action_dim（1行）
2. 修改action_bound（1行）
3. 修改action_all_training（1行）
4. 添加use_gat参数（1行）

**总共约4行代码修改**

### Step 3: 添加GAT开关（可选）

1. 在arguments.py添加参数（3行）
2. 在PPO_brain_AC.py添加GAT网络（复用现有代码）

---

## 📊 修改对比

| 文件 | 原始行数 | 新增行数 | 修改行数 | 总行数 |
|------|---------|---------|---------|--------|
| PPO_brain_AC.py | 301 | ~15 | ~5 | ~320 |
| main_PPO_AC.py | 301 | ~4 | ~4 | ~305 |
| arguments.py | - | ~3 | - | - |
| **总计** | 602 | **~22** | **~9** | **~625** |

**代码增加**: 不到4%

---

## 🎯 保持不变的部分

1. ✅ 训练流程（simulate → sample → train）
2. ✅ GAE计算
3. ✅ 奖励归一化
4. ✅ 联邦学习逻辑
5. ✅ TensorBoard日志
6. ✅ 模型保存/加载

---

## 🔄 修改后的数据流

### 原始流程
```
State → PPO → [RB, Power] → Environment → Reward
```

### 修改后流程
```
State → PPO → [RB, Power, Rho] → Environment (Semantic) → Reward
```

**变化**: 只增加一个输出维度

---

## 📝 实施顺序

1. **先不加GAT**: 只添加rho输出，使用原始的MLP网络
2. **验证训练**: 确保训练流程正常
3. **后续添加GAT**: 作为可选开关

这样可以逐步验证，降低风险。

