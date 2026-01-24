import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import copy

from arguments import get_args
args = get_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.set_random_seed)


my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth = True

n_hidden_1 = args.n_hidden_1
n_hidden_2 = args.n_hidden_2
n_hidden_3 = args.n_hidden_3

sigma_add = args.sigma_add

# 注意：本版本已移除 GAT 编码器实现（`graph_attention_layer` / `multi_layer_gat`）。
# 如需恢复 GAT，请回退到包含 GAT 的版本或重新引入相关实现。

class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, meta_lr, K, n_veh, n_RB, IS_meta, meta_episode,
                 use_gat=False, num_gat_heads=4, node_feature_dim=None):
        """
        Args:
            s_dim: state dimension (for backward compatibility, if use_gat=False)
            a_bound: action bounds [RB_bound, power_bound, compression_bound]
            c1, c2, epsilon, lr, meta_lr, K: PPO hyperparameters
            n_veh: number of UAVs (nodes in graph)
            n_RB: number of resource blocks
            IS_meta: whether to use meta learning
            meta_episode: meta episode number
            use_gat: whether to use GAT instead of MLP
            num_gat_heads: number of attention heads in GAT
            node_feature_dim: dimension of node features (CSI, location, etc.)
        """
        self.a_bound = a_bound
        self.K = K
        self.s_dim = s_dim  # Keep for backward compatibility
        self.a_dim = 3  # RB_choice + Power + Compression Ratio (rho)
        self.n_RB = n_RB
        self.n_veh = n_veh
        self.IS_meta = IS_meta
        self.meta_episode = meta_episode  # Save meta_episode as instance variable
        self.gamma = args.gamma
        self.GAE_discount = args.lambda_advantage
        if use_gat:
            raise ValueError("当前版本已移除 GAT 编码器（use_gat=True 不再支持）。")
        self.use_gat = False
        self.c1 = c1  # Value function loss weight
        self.c2 = c2  # Entropy weight
        self.epsilon = epsilon  # PPO clipping parameter
        
        # MLP-based input
        self.s_input = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')

        power_dist, RB_distribution, rho_distribution, self.v, params, self.saver = self._build_net('network', True)
        old_power_dist, old_RB_distribution, old_rho_distribution, old_v, old_params, _ = self._build_net('old_network', False)
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.v_pred_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_pred_next')
        self.gae = tf.placeholder(dtype=tf.float32, shape=[None], name='gae')

        GAE_advantage = self.gae

        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        RB_action = self.a[:,0]
        power_action = self.a[:,1]
        rho_action = self.a[:,2]  # Compression ratio
        
        # Beta分布处理：power和rho都使用Beta分布，输出[0,1]
        # Power需要从[-bound, bound]映射到[0,1]来计算概率
        # 映射公式：power_normalized = (power_action + action_bound[1]) / (2 * action_bound[1])
        power_normalized = (power_action + self.a_bound[1]) / (2 * self.a_bound[1] + 1e-8)
        power_normalized = tf.clip_by_value(power_normalized, 1e-6, 1.0 - 1e-6)  # 避免边界值
        power_prob = power_dist.prob(power_normalized)
        old_power_prob = old_power_dist.prob(power_normalized)
        
        # Rho已经是[0,1]范围，直接使用Beta分布
        rho_action_clipped = tf.clip_by_value(rho_action, 1e-6, 1.0 - 1e-6)  # 避免边界值
        rho_prob = rho_distribution.prob(rho_action_clipped)
        old_rho_prob = old_rho_distribution.prob(rho_action_clipped)
        
        # Original code style: simple ratio calculation with numerical stability
        # Add small epsilon to prevent division by zero
        ratio_power = power_prob / (old_power_prob + 1e-8)
        ratio_rho = rho_prob / (old_rho_prob + 1e-8)
        # Clip ratios to prevent extreme values
        ratio_power = tf.clip_by_value(ratio_power, 1e-6, 1e6)
        ratio_rho = tf.clip_by_value(ratio_rho, 1e-6, 1e6)
        
        # Value function loss
        v_pred = tf.squeeze(self.v, axis=-1) if len(self.v.get_shape()) > 1 else self.v
        
        # Clip values to prevent extreme values
        v_target = self.reward + self.gamma * self.v_pred_next
        v_target = tf.clip_by_value(v_target, -10.0, 10.0)
        v_pred = tf.clip_by_value(v_pred, -10.0, 10.0)
        L_vf = tf.reduce_mean(tf.square(v_target - v_pred))
        # Replace NaN with zeros
        L_vf = tf.where(tf.is_finite(L_vf), L_vf, tf.zeros_like(L_vf))
        
        # Use GAE_advantage directly (original code style)
        GAE_advantage_clipped = GAE_advantage
        
        # PPO clipping loss for power action
        L_clip_power = tf.reduce_mean(tf.minimum(
            ratio_power * GAE_advantage_clipped,
            tf.clip_by_value(ratio_power, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_clip_power = tf.where(tf.is_finite(L_clip_power), L_clip_power, tf.zeros_like(L_clip_power))
        
        # PPO clipping loss for RB action
        RB_prob = RB_distribution.prob(RB_action)
        old_RB_prob = old_RB_distribution.prob(RB_action)
        
        # Original code style: simple ratio calculation with numerical stability
        ratio_RB = RB_prob / (old_RB_prob + 1e-8)
        # Clip ratio to prevent extreme values
        ratio_RB = tf.clip_by_value(ratio_RB, 1e-6, 1e6)
        
        L_RB = tf.reduce_mean(tf.minimum(
            ratio_RB * GAE_advantage_clipped,
            tf.clip_by_value(ratio_RB, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_RB = tf.where(tf.is_finite(L_RB), L_RB, tf.zeros_like(L_RB))
        
        # PPO clipping loss for compression ratio (rho) action
        L_rho = tf.reduce_mean(tf.minimum(
            ratio_rho * GAE_advantage_clipped,
            tf.clip_by_value(ratio_rho, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_rho = tf.where(tf.is_finite(L_rho), L_rho, tf.zeros_like(L_rho))
        
        # Entropy for exploration
        S = tf.reduce_mean(power_dist.entropy() + RB_distribution.entropy() + rho_distribution.entropy())
        
        # Total loss (original code style, but with rho added)
        L = L_clip_power + L_RB + L_rho - c1 * L_vf + c2 * S
        # Replace NaN with zeros
        L = tf.where(tf.is_finite(L), L, tf.zeros_like(L))
        
        self.Loss = [L_clip_power, L_RB, L_rho, L_vf, S]
        self.Entropy_value = S
        
        # Sample actions (MLP mode)
        # Beta分布输出[0,1]，power需要映射到[-bound, bound]
        power_sample_beta = tf.squeeze(power_dist.sample(1), axis=0)  # [0,1]
        # 映射到[-bound, bound]：power_scaled = power_sample * 2 * bound - bound
        power_sample_scaled = power_sample_beta * 2 * self.a_bound[1] - self.a_bound[1]
        
        rho_sample = tf.squeeze(rho_distribution.sample(1), axis=0)  # [0,1]，Beta分布天然有界
        
        self.choose_action_op = tf.concat([
            tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)),
            power_sample_scaled,
            rho_sample
        ], 1)

        # Optimizer (original code style)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-L)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        self.sesses = []
        for ind_agent in range(self.n_veh):
            print("Initializing agent", ind_agent)
            sess = tf.Session(config=my_config)
            sess.run(tf.global_variables_initializer())
            self.sesses.append(sess)
        if self.IS_meta:
            print("\nRestoring the model...")
            # 使用SEE优化目标（与meta_train_PPO_AC.py保持一致）
            optimization_target = 'SEE'
            
            # Meta训练保存格式: AC_SEE_{sigma_add}_{meta_episode}_{lr_meta_a}
            # 与backup版本的区别：backup版本格式为 AC_{sigma_add}_{meta_episode}_{lr_meta_a}（无SEE）
            # 当前版本格式为 AC_SEE_{sigma_add}_{meta_episode}_{lr_meta_a}（有SEE）
            opt_suffix = optimization_target  # 直接使用'SEE'
            
            # 使用实例变量self.meta_episode和args中的参数
            for i in range(self.n_veh):
                meta_save_path = args.save_path  # 使用args.save_path，默认是'meta_model_'
                model_path = meta_save_path + 'AC_' + opt_suffix + '_' + '%s_' %sigma_add + '%d_' % self.meta_episode +'%s_' %args.lr_meta_a
                print(f"Loading meta model for agent {i}: {model_path}")
                print(f"  Expected path format: {model_path}")
                self.load_models(self.sesses[i], model_path, self.saver)

    def load_models(self, sess, model_path, saver):
        """ Restore models from the current directory with the name filename """
        dir_ = os.path.dirname(os.path.realpath(__file__))
        full_model_path = os.path.join(dir_, "model/" + model_path)
        
        # Check if model file exists
        if os.path.exists(full_model_path + '.index'):
            try:
                saver.restore(sess, full_model_path)
                print(f"✅ Successfully loaded model: {full_model_path}")
            except Exception as e:
                error_str = str(e)
                # 检查是否是形状不匹配错误（旧模型没有rho参数）
                if "shape mismatch" in error_str.lower() or "Assign requires shapes" in error_str:
                    print(f"⚠️  Shape mismatch detected!")
                    print(f"   The saved meta model was trained with OLD code (without rho/compression ratio parameters).")
                    print(f"   Current code requires rho parameters, so the model cannot be loaded.")
                    print(f"")
                    print(f"   SOLUTION: Re-train the meta model with current code:")
                    print(f"   python meta_train_PPO_AC.py --n_veh_list 2,4,8 --n_RB 10 --sigma_add 0.3 --meta_episode 100 --lr_meta_a 5e-7 --lr_meta_c 1e-5")
                    print(f"")
                    print(f"   Continuing with randomly initialized weights (no meta model loaded)...")
                else:
                    print(f"❌ Failed to load model: {full_model_path}")
                    print(f"   Error: {e}")
                    print(f"   Please re-train the meta model using current code.")
                    print(f"   Continuing with randomly initialized weights...")
        else:
            print(f"⚠️  Warning: Model file does not exist: {full_model_path}")
            print(f"   Looking for: {full_model_path}.index")
            
            # 尝试兼容backup版本的路径格式（无SEE后缀）
            if 'AC_SEE_' in model_path:
                backup_model_path = model_path.replace('AC_SEE_', 'AC_')
                backup_full_path = os.path.join(dir_, "model/" + backup_model_path)
                if os.path.exists(backup_full_path + '.index'):
                    try:
                        saver.restore(sess, backup_full_path)
                        print(f"✅ Successfully loaded model (backup format): {backup_full_path}")
                        return
                    except Exception as e:
                        print(f"   Also tried backup format but failed: {backup_full_path}")
            
            print(f"   Please train the meta model first:")
            print(f"   python meta_train_PPO_AC.py --n_veh_list 2,4,8 --n_RB 10 --sigma_add 0.3 --meta_episode 100 --lr_meta_a 5e-7 --lr_meta_c 1e-5")
            print(f"   Continuing with randomly initialized weights...")

    def save_model(self, sess, model_path, saver):
        """ Save models to the current directory with the name filename """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        saver.save(sess, model_path, write_meta_graph=False)

    def save_models(self, label):
        for i in range(self.n_veh):
            model_path = label + '/agent_' + str(i)
            self.save_model(self.sesses[i], model_path, self.saver)
            
    def _build_net(self, scope, trainable):
        """
        Build network (MLP encoder) with multiple actor heads
        Returns: (power_dist, RB_dist, rho_dist, value, params, saver)
        """
        with tf.variable_scope(scope):
            initializer = tf.compat.v1.keras.initializers.he_normal()

            # MLP encoder
            # 注意：变量创建顺序必须与meta_brain_PPO.py完全一致，否则无法加载meta模型
            # 顺序：所有weights先创建，然后所有biases
            self.w_1 = tf.Variable(initializer(shape=(self.s_dim, n_hidden_1)), trainable=trainable)
            self.w_2 = tf.Variable(initializer(shape=(n_hidden_1, n_hidden_2)), trainable=trainable)
            self.w_3 = tf.Variable(initializer(shape=(n_hidden_2, n_hidden_3)), trainable=trainable)
            # Power Beta分布参数：alpha和beta
            self.w_power_alpha = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_power_beta = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_RB = tf.Variable(initializer(shape=(n_hidden_2, self.n_RB)), trainable=trainable)
            # Rho Beta分布参数：alpha和beta
            self.w_rho_alpha = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_rho_beta = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_v = tf.Variable(initializer(shape=(n_hidden_3, 1)), trainable=trainable)

            self.b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1), trainable=trainable)
            self.b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1), trainable=trainable)
            self.b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1), trainable=trainable)
            # Power Beta分布bias
            self.b_power_alpha = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_power_beta = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_RB = tf.Variable(tf.truncated_normal([self.n_RB], stddev=0.1), trainable=trainable)
            # Rho Beta分布bias
            self.b_rho_alpha = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_rho_beta = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_v = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)

            layer_p1 = tf.nn.relu(tf.add(tf.matmul(self.s_input, self.w_1), self.b_1), name='p_1')
            layer_1_b = tf.layers.batch_normalization(layer_p1)
            layer_p2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, self.w_2), self.b_2), name='p_2')
            layer_2_b = tf.layers.batch_normalization(layer_p2)
            layer_p3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, self.w_3), self.b_3), name='p_3')
            layer_3_b = tf.layers.batch_normalization(layer_p3)

            critic_input = layer_3_b  # For critic
            
            # Actor heads
            RB_probs = tf.nn.softmax(tf.add(tf.matmul(layer_2_b, self.w_RB), self.b_RB), name='RB_layer')
            RB_distribution = tf.distributions.Categorical(probs=RB_probs)
            
            # Power Beta分布：alpha和beta参数（确保>1以避免极端值）
            power_alpha_raw = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_power_alpha), self.b_power_alpha), name='power_alpha_layer')
            power_beta_raw = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_power_beta), self.b_power_beta), name='power_beta_layer')
            power_alpha = power_alpha_raw + 1.0  # 确保alpha > 1
            power_beta = power_beta_raw + 1.0    # 确保beta > 1
            power_distribution = tf.distributions.Beta(concentration1=power_alpha, concentration0=power_beta)
            
            # Rho Beta分布：alpha和beta参数（确保>1以避免极端值）
            rho_alpha_raw = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_alpha), self.b_rho_alpha), name='rho_alpha_layer')
            rho_beta_raw = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_beta), self.b_rho_beta), name='rho_beta_layer')
            rho_alpha = rho_alpha_raw + 1.0  # 确保alpha > 1
            rho_beta = rho_beta_raw + 1.0    # 确保beta > 1
            rho_distribution = tf.distributions.Beta(concentration1=rho_alpha, concentration0=rho_beta)
            
            # Critic network (MLP)
            v = tf.nn.relu(tf.add(tf.matmul(critic_input, self.w_v), self.b_v), name='v_layer')
            
            saver = tf.train.Saver(max_to_keep=self.n_veh * 2)
            
        params = tf.global_variables(scope)
        return power_distribution, RB_distribution, rho_distribution, v, params, saver

    def get_v(self, s, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Get value function estimate
        Args:
            s: state (for backward compatibility with MLP)
            sess: TensorFlow session
            node_features: [n_veh, node_feature_dim] node features (for GAT)
            adj_matrix: [n_veh, n_veh] adjacency matrix (for GAT)
            agent_idx: agent index (for GAT mode, to select which node's value)
        """
        if node_features is not None or adj_matrix is not None:
            raise ValueError("当前版本已移除 GAT 编码器：get_v 不再支持 node_features/adj_matrix 输入。")
        return sess.run(self.v, {self.s_input: np.array([s])}).squeeze()

    def choose_action(self, s, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Choose action
        Args:
            s: state (for backward compatibility with MLP)
            sess: TensorFlow session
            node_features: [n_veh, node_feature_dim] node features (for GAT)
            adj_matrix: [n_veh, n_veh] adjacency matrix (for GAT)
            agent_idx: agent index (for GAT mode, to select which node's action)
        """
        if node_features is not None or adj_matrix is not None:
            raise ValueError("当前版本已移除 GAT 编码器：choose_action 不再支持 node_features/adj_matrix 输入。")
        a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]  # RB (Categorical)
        # Power已经是Beta分布映射后的值，在[-bound, bound]范围内，但为了安全还是clip一下
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        # Rho是Beta分布输出，天然在[0,1]范围内，但为了安全还是clip一下
        clipped_a[2] = np.clip(a[2], 0.0, 1.0)
        return clipped_a

    def train(self, s, a, gae, reward, v_pred_next, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Train the network
        Args:
            s: state (for backward compatibility with MLP)
            a: actions [batch_size, 3] (RB, Power, Compression Ratio)
            gae: GAE advantages
            reward: rewards
            v_pred_next: next state values
            sess: TensorFlow session
            node_features: [batch_size, n_veh, node_feature_dim] (for GAT)
            adj_matrix: [batch_size, n_veh, n_veh] (for GAT)
        """
        sess.run(self.update_params_op)
        
        if node_features is not None or adj_matrix is not None:
            raise ValueError("当前版本已移除 GAT 编码器：train 不再支持 node_features/adj_matrix 输入。")
        feed_dict = {
            self.s_input: s,
            self.a: a,
            self.reward: reward,
            self.v_pred_next: v_pred_next,
            self.gae: gae
        }
        
        # K epochs
        for i in range(self.K):
            sess.run(self.train_op, feed_dict)
        
        # Get loss components for debugging
        loss_components = sess.run(self.Loss, feed_dict)
        entropy = sess.run(self.Entropy_value, feed_dict)
        
        return [loss_components, entropy]

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t) - can be [T, n_veh] or [T]
        :param v_preds: v(st) - can be [T, n_veh] or [T]
        :param v_preds_next: v(st+1) - can be [T, n_veh] or [T]
        :return: gaes - same shape as input
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Handle 2D case: [T, n_veh]
        if len(rewards.shape) == 2:
            # For each agent, compute GAE separately
            n_veh = rewards.shape[1]
            gaes_list = []
            for agent_idx in range(n_veh):
                r_agent = rewards[:, agent_idx]
                v_agent = v_preds[:, agent_idx]
                v_next_agent = v_preds_next[:, agent_idx]
                
                # Compute deltas for this agent
                deltas = r_agent + self.gamma * v_next_agent - v_agent
                gaes = deltas.copy()
                
                # Compute GAE backwards
                for t in reversed(range(len(gaes) - 1)):
                    gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
                
                gaes_list.append(gaes)
            
            # Stack back to [T, n_veh]
            return np.stack(gaes_list, axis=1)
        else:
            # 1D case (MLP mode: [T])
            deltas = rewards + self.gamma * v_preds_next - v_preds
            gaes = deltas.copy()
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
            return gaes

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t) - can be [T, n_veh] or [T]
        :param v_preds: v(st) - can be [T, n_veh] or [T]
        :param v_preds_next: v(st+1) - can be [T, n_veh] or [T]
        :return: gaes - same shape as input
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Handle 2D case (GAT mode: [T, n_veh])
        if len(rewards.shape) == 2:
            # For each agent, compute GAE separately
            n_veh = rewards.shape[1]
            gaes_list = []
            for agent_idx in range(n_veh):
                r_agent = rewards[:, agent_idx]
                v_agent = v_preds[:, agent_idx]
                v_next_agent = v_preds_next[:, agent_idx]
                
                # Compute deltas for this agent
                deltas = r_agent + self.gamma * v_next_agent - v_agent
                gaes = deltas.copy()
                
                # Compute GAE backwards
                for t in reversed(range(len(gaes) - 1)):
                    gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
                
                gaes_list.append(gaes)
            
            # Stack back to [T, n_veh]
            return np.stack(gaes_list, axis=1)
        else:
            # 1D case (MLP mode: [T])
            deltas = rewards + self.gamma * v_preds_next - v_preds
            gaes = deltas.copy()
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
            return gaes

    def averaging_model(self, success_rate, aggregation_weight=1.0):
        """
        联邦学习模型聚合
        Args:
            success_rate: 各UE的成功率，用于计算聚合权重
            aggregation_weight: 聚合权重（0.0-1.0）
                - 1.0: 硬替换（完全使用聚合参数，原有逻辑）
                - 0.7: 软聚合（70%聚合参数 + 30%本地参数）
                - 0.0: 不聚合（仅用于测试）
        """
        # 仅保留 MLP 模式下的手工参数平均（联邦学习）
        # 注意：已更新为Beta分布参数
        # 改进：使用基于success_rate的加权聚合，而不是简单平均
        # 改进：支持软聚合（部分替换），保留部分本地参数
        sigma = 1e-8
        
        # 确保aggregation_weight在有效范围内
        aggregation_weight = np.clip(aggregation_weight, 0.0, 1.0)
        
        # 处理success_rate：转换为权重
        if success_rate is not None and len(success_rate) > 0:
            # 将success_rate归一化为权重（避免除零）
            success_rate = np.array(success_rate)
            success_rate = np.clip(success_rate, 0.0, 1.0)  # 确保在[0,1]范围内
            # 添加小的epsilon避免全零情况
            weights = success_rate + 1e-6
            weights = weights / weights.sum()  # 归一化
        else:
            # 如果没有success_rate，使用均匀权重
            weights = np.ones(self.n_veh) / self.n_veh
        
        w_1_mean = np.random.normal(0, sigma, [self.s_dim, n_hidden_1])
        w_2_mean = np.random.normal(0, sigma, [n_hidden_1, n_hidden_2])
        w_3_mean = np.random.normal(0, sigma, [n_hidden_2, n_hidden_3])
        # Power Beta分布参数
        w_power_alpha_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_power_beta_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_RB_mean = np.random.normal(0, sigma, [n_hidden_2, self.n_RB])
        # Rho Beta分布参数
        w_rho_alpha_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_rho_beta_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_v_mean = np.random.normal(0, sigma, [n_hidden_3, 1])

        b_1_mean = np.random.normal(0, sigma, [n_hidden_1])
        b_2_mean = np.random.normal(0, sigma, [n_hidden_2])
        b_3_mean = np.random.normal(0, sigma, [n_hidden_3])
        # Power Beta分布bias
        b_power_alpha_mean = np.random.normal(0, sigma, [1])
        b_power_beta_mean = np.random.normal(0, sigma, [1])
        b_RB_mean = np.random.normal(0, sigma, [self.n_RB])
        # Rho Beta分布bias
        b_rho_alpha_mean = np.random.normal(0, sigma, [1])
        b_rho_beta_mean = np.random.normal(0, sigma, [1])
        b_v_mean = np.random.normal(0, sigma, [1])

        # 使用加权聚合（基于success_rate）
        for i in range(self.n_veh):
            weight = weights[i]
            w_1_mean += self.sesses[i].run(self.w_1) * weight
            w_2_mean += self.sesses[i].run(self.w_2) * weight
            w_3_mean += self.sesses[i].run(self.w_3) * weight
            w_power_alpha_mean += self.sesses[i].run(self.w_power_alpha) * weight
            w_power_beta_mean += self.sesses[i].run(self.w_power_beta) * weight
            w_RB_mean += self.sesses[i].run(self.w_RB) * weight
            w_rho_alpha_mean += self.sesses[i].run(self.w_rho_alpha) * weight
            w_rho_beta_mean += self.sesses[i].run(self.w_rho_beta) * weight
            w_v_mean += self.sesses[i].run(self.w_v) * weight

            b_1_mean += self.sesses[i].run(self.b_1) * weight
            b_2_mean += self.sesses[i].run(self.b_2) * weight
            b_3_mean += self.sesses[i].run(self.b_3) * weight
            b_power_alpha_mean += self.sesses[i].run(self.b_power_alpha) * weight
            b_power_beta_mean += self.sesses[i].run(self.b_power_beta) * weight
            b_RB_mean += self.sesses[i].run(self.b_RB) * weight
            b_rho_alpha_mean += self.sesses[i].run(self.b_rho_alpha) * weight
            b_rho_beta_mean += self.sesses[i].run(self.b_rho_beta) * weight
            b_v_mean += self.sesses[i].run(self.b_v) * weight

        # 软聚合：混合聚合参数和本地参数
        for i in range(self.n_veh):
            if aggregation_weight < 1.0:
                # 软聚合：保留部分本地参数
                # 获取当前本地参数
                old_w_1 = self.sesses[i].run(self.w_1)
                old_w_2 = self.sesses[i].run(self.w_2)
                old_w_3 = self.sesses[i].run(self.w_3)
                old_w_power_alpha = self.sesses[i].run(self.w_power_alpha)
                old_w_power_beta = self.sesses[i].run(self.w_power_beta)
                old_w_RB = self.sesses[i].run(self.w_RB)
                old_w_rho_alpha = self.sesses[i].run(self.w_rho_alpha)
                old_w_rho_beta = self.sesses[i].run(self.w_rho_beta)
                old_w_v = self.sesses[i].run(self.w_v)
                
                old_b_1 = self.sesses[i].run(self.b_1)
                old_b_2 = self.sesses[i].run(self.b_2)
                old_b_3 = self.sesses[i].run(self.b_3)
                old_b_power_alpha = self.sesses[i].run(self.b_power_alpha)
                old_b_power_beta = self.sesses[i].run(self.b_power_beta)
                old_b_RB = self.sesses[i].run(self.b_RB)
                old_b_rho_alpha = self.sesses[i].run(self.b_rho_alpha)
                old_b_rho_beta = self.sesses[i].run(self.b_rho_beta)
                old_b_v = self.sesses[i].run(self.b_v)
                
                # 软聚合：混合新旧参数
                # new_param = aggregation_weight * aggregated_param + (1 - aggregation_weight) * local_param
                self.sesses[i].run(self.w_1.assign(aggregation_weight * w_1_mean + (1 - aggregation_weight) * old_w_1))
                self.sesses[i].run(self.w_2.assign(aggregation_weight * w_2_mean + (1 - aggregation_weight) * old_w_2))
                self.sesses[i].run(self.w_3.assign(aggregation_weight * w_3_mean + (1 - aggregation_weight) * old_w_3))
                self.sesses[i].run(self.w_power_alpha.assign(aggregation_weight * w_power_alpha_mean + (1 - aggregation_weight) * old_w_power_alpha))
                self.sesses[i].run(self.w_power_beta.assign(aggregation_weight * w_power_beta_mean + (1 - aggregation_weight) * old_w_power_beta))
                self.sesses[i].run(self.w_RB.assign(aggregation_weight * w_RB_mean + (1 - aggregation_weight) * old_w_RB))
                self.sesses[i].run(self.w_rho_alpha.assign(aggregation_weight * w_rho_alpha_mean + (1 - aggregation_weight) * old_w_rho_alpha))
                self.sesses[i].run(self.w_rho_beta.assign(aggregation_weight * w_rho_beta_mean + (1 - aggregation_weight) * old_w_rho_beta))
                self.sesses[i].run(self.w_v.assign(aggregation_weight * w_v_mean + (1 - aggregation_weight) * old_w_v))
                
                self.sesses[i].run(self.b_1.assign(aggregation_weight * b_1_mean + (1 - aggregation_weight) * old_b_1))
                self.sesses[i].run(self.b_2.assign(aggregation_weight * b_2_mean + (1 - aggregation_weight) * old_b_2))
                self.sesses[i].run(self.b_3.assign(aggregation_weight * b_3_mean + (1 - aggregation_weight) * old_b_3))
                self.sesses[i].run(self.b_power_alpha.assign(aggregation_weight * b_power_alpha_mean + (1 - aggregation_weight) * old_b_power_alpha))
                self.sesses[i].run(self.b_power_beta.assign(aggregation_weight * b_power_beta_mean + (1 - aggregation_weight) * old_b_power_beta))
                self.sesses[i].run(self.b_RB.assign(aggregation_weight * b_RB_mean + (1 - aggregation_weight) * old_b_RB))
                self.sesses[i].run(self.b_rho_alpha.assign(aggregation_weight * b_rho_alpha_mean + (1 - aggregation_weight) * old_b_rho_alpha))
                self.sesses[i].run(self.b_rho_beta.assign(aggregation_weight * b_rho_beta_mean + (1 - aggregation_weight) * old_b_rho_beta))
                self.sesses[i].run(self.b_v.assign(aggregation_weight * b_v_mean + (1 - aggregation_weight) * old_b_v))
            else:
                # 硬替换（原有逻辑）：完全使用聚合参数
                self.sesses[i].run(self.w_1.assign(w_1_mean))
                self.sesses[i].run(self.w_2.assign(w_2_mean))
                self.sesses[i].run(self.w_3.assign(w_3_mean))
                self.sesses[i].run(self.w_power_alpha.assign(w_power_alpha_mean))
                self.sesses[i].run(self.w_power_beta.assign(w_power_beta_mean))
                self.sesses[i].run(self.w_RB.assign(w_RB_mean))
                self.sesses[i].run(self.w_rho_alpha.assign(w_rho_alpha_mean))
                self.sesses[i].run(self.w_rho_beta.assign(w_rho_beta_mean))
                self.sesses[i].run(self.w_v.assign(w_v_mean))
                
                self.sesses[i].run(self.b_1.assign(b_1_mean))
                self.sesses[i].run(self.b_2.assign(b_2_mean))
                self.sesses[i].run(self.b_3.assign(b_3_mean))
                self.sesses[i].run(self.b_power_alpha.assign(b_power_alpha_mean))
                self.sesses[i].run(self.b_power_beta.assign(b_power_beta_mean))
                self.sesses[i].run(self.b_RB.assign(b_RB_mean))
                self.sesses[i].run(self.b_rho_alpha.assign(b_rho_alpha_mean))
                self.sesses[i].run(self.b_rho_beta.assign(b_rho_beta_mean))
                self.sesses[i].run(self.b_v.assign(b_v_mean))
