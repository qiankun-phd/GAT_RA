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
fl_noise_sigma = getattr(args, 'fl_noise_sigma', 1e-8)

# 注意：本版本已移除 GAT 编码器实现（`graph_attention_layer` / `multi_layer_gat`）。
# 如需恢复 GAT，请回退到包含 GAT 的版本或重新引入相关实现。

class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, meta_lr, K, n_veh, n_RB, IS_meta, meta_episode):
        self.a_bound = a_bound
        self.K = K
        self.s_dim = s_dim
        self.a_dim = 3  # RB_choice + Power + rho
        self.n_RB = n_RB
        self.n_veh = n_veh
        self.IS_meta = IS_meta
        self.meta_episode = meta_episode
        self.gamma = args.gamma
        self.GAE_discount = args.lambda_advantage
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
        rho_action = self.a[:,2]

        ratio_power = power_dist.prob(power_action) / old_power_dist.prob(power_action)
        ratio_rho = rho_distribution.prob(rho_action) / old_rho_distribution.prob(rho_action)
        L_vf = tf.reduce_mean(tf.square(self.reward + self.gamma * self.v_pred_next - self.v))

        L_clip_power = tf.reduce_mean(tf.minimum(
            ratio_power * GAE_advantage,
            tf.clip_by_value(ratio_power, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))
        ratio_RB = RB_distribution.prob(RB_action) / old_RB_distribution.prob(RB_action)
        L_RB = tf.reduce_mean(tf.minimum(
            ratio_RB * GAE_advantage,
            tf.clip_by_value(ratio_RB, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))
        L_rho = tf.reduce_mean(tf.minimum(
            ratio_rho * GAE_advantage,
            tf.clip_by_value(ratio_rho, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))
        S = tf.reduce_mean(power_dist.entropy() + RB_distribution.entropy() + rho_distribution.entropy())

        L = L_clip_power + L_RB + L_rho - c1 * L_vf + c2 * S
        self.Loss = [L_clip_power, L_RB, L_rho, L_vf, S]
        self.Entropy_value = S
        power_sample = tf.squeeze(power_dist.sample(1), axis=0)
        rho_sample = tf.squeeze(rho_distribution.sample(1), axis=0)
        self.choose_action_op = tf.concat([
            tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)),
            power_sample,
            rho_sample
        ], 1)

        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='lr')
        self.lr_default = lr
        self.train_op = tf.train.AdamOptimizer(self.lr_ph).minimize(-L)
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
            opt_suffix = optimization_target  # 直接使用'SEE'
            area_size = getattr(args, 'area_size', 25.0)

            # Meta路径：支持 --meta_model_path 覆盖，否则按 area 自动构建
            # 格式: meta_model_AC_SEE_{sigma_add}_{meta_episode}_{lr_meta_a}_area{area}_
            if getattr(args, 'meta_model_path', ''):
                model_path = args.meta_model_path
                print(f"Using override meta_model_path: {model_path}")
            else:
                model_path = args.save_path + 'AC_' + opt_suffix + '_' + '%s_' % sigma_add + '%d_' % self.meta_episode + '%s_' % args.lr_meta_a + 'area%d_' % int(area_size)

            for i in range(self.n_veh):
                print(f"Loading meta model for agent {i}: {model_path}")
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
                    print(f"   python meta_train_PPO_AC.py --n_veh_list 2,4,8 --n_RB 10 --sigma_add 0.1 --meta_episode 100 --lr_meta_a 5e-7 --lr_meta_c 1e-5")
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

            # 尝试无 area 后缀的路径（旧版 meta 模型）
            if '_area' in model_path:
                fallback_path = model_path.split('_area')[0] + '_'
                fallback_full = os.path.join(dir_, "model/" + fallback_path)
                if os.path.exists(fallback_full + '.index'):
                    try:
                        saver.restore(sess, fallback_full)
                        print(f"✅ Loaded meta model (no area suffix): {fallback_full}")
                        return
                    except Exception as e:
                        print(f"   Fallback (no area) also failed: {fallback_full}")

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
            print(f"   python meta_train_PPO_AC.py --n_veh_list 2,4,8 --n_RB 10 --sigma_add 0.1 --meta_episode 100 --lr_meta_a 5e-7 --lr_meta_c 1e-5")
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

    def load_agents_from_label(self, label):
        """Load trained agents from directory label (label/agent_0, ...). Policy is shared so load agent_0 into all sessions."""
        dir_ = os.path.dirname(os.path.realpath(__file__))
        base = os.path.join(dir_, "model", label)
        if not os.path.exists(base + '/agent_0.index'):
            print(f"⚠️  No checkpoint at {base}/agent_0")
            return False
        model_path = label + '/agent_0'
        for i in range(self.n_veh):
            self.load_models(self.sesses[i], model_path, self.saver)
        return True

    def _build_net(self, scope, trainable):
        with tf.variable_scope(scope):
            initializer = tf.compat.v1.keras.initializers.he_normal()

            self.w_1 = tf.Variable(initializer(shape=(self.s_dim, n_hidden_1)), trainable=trainable)
            self.w_2 = tf.Variable(initializer(shape=(n_hidden_1, n_hidden_2)), trainable=trainable)
            self.w_3 = tf.Variable(initializer(shape=(n_hidden_2, n_hidden_3)), trainable=trainable)
            self.w_power_mu = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_power_sigma = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_RB = tf.Variable(initializer(shape=(n_hidden_2, self.n_RB)), trainable=trainable)
            self.w_rho_mu = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_rho_sigma = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_v = tf.Variable(initializer(shape=(n_hidden_3, 1)), trainable=trainable)

            self.b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1), trainable=trainable)
            self.b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1), trainable=trainable)
            self.b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1), trainable=trainable)
            self.b_power_mu = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_power_sigma = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_RB = tf.Variable(tf.truncated_normal([self.n_RB], stddev=0.1), trainable=trainable)
            self.b_rho_mu = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_rho_sigma = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_v = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)

            layer_p1 = tf.nn.relu(tf.add(tf.matmul(self.s_input, self.w_1), self.b_1), name='p_1')
            layer_1_b = tf.layers.batch_normalization(layer_p1)
            layer_p2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, self.w_2), self.b_2), name='p_2')
            layer_2_b = tf.layers.batch_normalization(layer_p2)
            layer_p3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, self.w_3), self.b_3), name='p_3')
            layer_3_b = tf.layers.batch_normalization(layer_p3)

            power_mu = tf.nn.tanh(tf.add(tf.matmul(layer_2_b, self.w_power_mu), self.b_power_mu), name='power_mu_layer')
            power_sigma = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_power_sigma), self.b_power_sigma), name='power_sigma_layer')
            RB_probs = tf.nn.softmax(tf.add(tf.matmul(layer_2_b, self.w_RB), self.b_RB), name='RB_layer')
            RB_distribution = tf.distributions.Categorical(probs=RB_probs)
            rho_mu = tf.nn.tanh(tf.add(tf.matmul(layer_2_b, self.w_rho_mu), self.b_rho_mu), name='rho_mu_layer')
            rho_sigma = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_sigma), self.b_rho_sigma), name='rho_sigma_layer')

            saver = tf.train.Saver(max_to_keep=self.n_veh * 2)
            v = tf.nn.relu(tf.add(tf.matmul(layer_3_b, self.w_v), self.b_v), name='v_layer')

            power_mu, power_sigma = power_mu, power_sigma + sigma_add
            rho_mu, rho_sigma = rho_mu, rho_sigma + sigma_add
            power_distribution = tf.distributions.Normal(loc=power_mu, scale=power_sigma)
            rho_distribution = tf.distributions.Normal(loc=rho_mu, scale=rho_sigma)
        params = tf.global_variables(scope)
        return power_distribution, RB_distribution, rho_distribution, v, params, saver

    def get_v(self, s, sess):
        return sess.run(self.v, {self.s_input: np.array([s])}).squeeze()

    def choose_action(self, s, sess):
        a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]  # RB (Categorical)
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        clipped_a[2] = np.clip(a[2], -self.a_bound[2], self.a_bound[2])
        return clipped_a

    def train(self, s, a, gae, reward, v_pred_next, sess, lr=None):
        """
        Args:
            s: state [batch, s_dim]
            a: actions [batch, 3] (RB, Power, rho)
            gae: GAE advantages [batch]
            reward: rewards [batch]
            v_pred_next: next state values [batch]
            sess: TensorFlow session
            lr: learning rate (None = use default)
        """
        sess.run(self.update_params_op)
        lr_val = float(lr) if lr is not None else self.lr_default
        feed_dict = {
            self.s_input: s,
            self.a: a,
            self.reward: reward,
            self.v_pred_next: v_pred_next,
            self.gae: gae,
            self.lr_ph: lr_val
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

    def averaging_model(self, success_rate, aggregation_weight=1.0, layer_wise=False, external_weights=None,
                        use_success_rate_weighting=False):
        """
        联邦学习模型聚合
        Args:
            success_rate: 各UE的成功率（保留用于兼容性，但不再使用作为默认权重）
            aggregation_weight: 聚合权重（0.0-1.0）
                - 1.0: 硬替换（完全使用聚合参数，原有逻辑）
                - 0.7: 软聚合（70%聚合参数 + 30%本地参数）
                - 0.0: 不聚合（仅用于测试）
            layer_wise: 分层联邦聚合开关（默认False）
                - True: 只聚合特征提取层(w_1,w_2,b_1,b_2)，保留决策层个性化
                - False: 聚合所有网络参数（标准联邦学习）
            external_weights: 外部提供的聚合权重（如语义感知权重）
                - 如果提供，将使用此权重
                - 如果为None，根据use_success_rate_weighting决定
                - 应为归一化的numpy数组，长度为n_veh
            use_success_rate_weighting: 是否使用成功率权重（默认False）
                - True: 使用success_rate作为聚合权重
                - False: 使用均匀权重（标准FedAvg - 平均模型，默认）
        """
        # 仅保留 MLP 模式下的手工参数平均（联邦学习）
        # 注意：方案三为 Normal(mu,sigma) 参数
        # 默认使用均匀权重（标准FedAvg - 平均模型），而不是success_rate加权
        # 改进：支持软聚合（部分替换），保留部分本地参数
        # 与 backup 一致：聚合累加器默认用极小随机噪声初始化（sigma 由参数 fl_noise_sigma 控制）
        sigma = fl_noise_sigma
        
        # 确保aggregation_weight在有效范围内
        aggregation_weight = np.clip(aggregation_weight, 0.0, 1.0)
        
        # 打印分层聚合状态
        if layer_wise:
            print("🔄 分层联邦聚合: 只聚合特征提取层(w_1,w_2,b_1,b_2)，保留决策层个性化")
        else:
            print("🔄 标准联邦聚合: 聚合所有网络参数")
        
        # 处理权重：优先使用external_weights，否则根据use_success_rate_weighting决定
        # 默认使用均匀权重（标准FedAvg - 平均模型），而不是success_rate
        if external_weights is not None:
            # 使用外部提供的权重（如语义感知权重）
            weights = np.array(external_weights)
            # 确保权重归一化
            weights = weights / (weights.sum() + 1e-8)
            print(f"🔄 使用语义感知权重: {np.round(weights, 3)}")
        elif use_success_rate_weighting and success_rate is not None and len(success_rate) > 0:
            # 如果明确启用success_rate权重，使用成功率权重
            success_rate = np.array(success_rate)
            success_rate = np.clip(success_rate, 0.0, 1.0)  # 确保在[0,1]范围内
            # 添加小的epsilon避免全零情况
            weights = success_rate + 1e-6
            weights = weights / weights.sum()  # 归一化
            print(f"🔄 使用成功率权重: {np.round(weights, 3)}")
        else:
            # 默认使用均匀权重（标准FedAvg - 平均模型）
            weights = np.ones(self.n_veh) / self.n_veh
            print(f"🔄 使用均匀权重（平均模型）: {np.round(weights, 3)}")
        
        # 🔍 关键诊断：打印当前FL配置，确保配置正确传递
        print(f"📋 FL配置摘要:")
        print(f"   aggregation_weight={aggregation_weight:.3f} ({'软聚合' if aggregation_weight < 1.0 else '硬替换'})")
        print(f"   layer_wise={layer_wise} ({'分层聚合' if layer_wise else '标准聚合'})")
        print(f"   external_weights={'已提供' if external_weights is not None else 'None'}")
        print(f"   use_success_rate_weighting={use_success_rate_weighting}")
        print(f"   累加器初始化: 随机噪声 (sigma={sigma}, 与backup风格一致)")
        
        # 特征提取层 (Encoder) - 始终聚合
        # 所有FL配置默认使用backup风格的累加器初始化方式（随机噪声，sigma=fl_noise_sigma）
        w_1_mean = np.random.normal(0, sigma, [self.s_dim, n_hidden_1])
        w_2_mean = np.random.normal(0, sigma, [n_hidden_1, n_hidden_2])
        b_1_mean = np.random.normal(0, sigma, [n_hidden_1])
        b_2_mean = np.random.normal(0, sigma, [n_hidden_2])
        
        # 决策层 (Task-specific Heads) - 只有在非分层模式下才聚合
        if not layer_wise:
            w_3_mean = np.random.normal(0, sigma, [n_hidden_2, n_hidden_3])
            w_power_mu_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
            w_power_sigma_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
            w_RB_mean = np.random.normal(0, sigma, [n_hidden_2, self.n_RB])
            w_rho_mu_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
            w_rho_sigma_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
            w_v_mean = np.random.normal(0, sigma, [n_hidden_3, 1])
            
            b_3_mean = np.random.normal(0, sigma, [n_hidden_3])
            b_power_mu_mean = np.random.normal(0, sigma, [1])
            b_power_sigma_mean = np.random.normal(0, sigma, [1])
            b_RB_mean = np.random.normal(0, sigma, [self.n_RB])
            b_rho_mu_mean = np.random.normal(0, sigma, [1])
            b_rho_sigma_mean = np.random.normal(0, sigma, [1])
            b_v_mean = np.random.normal(0, sigma, [1])

        # 使用加权聚合（基于权重）
        # 验证：记录聚合前的参数，用于后续验证
        w_1_before_agg = None
        w_1_all_before = []  # 记录所有agent聚合前的参数
        if len(self.sesses) > 0:
            w_1_before_agg = self.sesses[0].run(self.w_1).copy()
            # 记录所有agent的参数
            for i in range(self.n_veh):
                w_1_all_before.append(self.sesses[i].run(self.w_1).copy())
        
        # 🔍 关键诊断：检查聚合前各agent的参数是否不同
        if len(w_1_all_before) >= 2:
            param_diffs_before = []
            for i in range(1, len(w_1_all_before)):
                diff = np.mean(np.abs(w_1_all_before[0] - w_1_all_before[i]))
                param_diffs_before.append(diff)
            max_diff_before = max(param_diffs_before) if param_diffs_before else 0.0
            min_diff_before = min(param_diffs_before) if param_diffs_before else 0.0
            avg_diff_before = np.mean(param_diffs_before) if param_diffs_before else 0.0
            
            print(f"🔍 聚合前参数差异检查（关键指标）:")
            print(f"   最大差异: {max_diff_before:.8f}")
            print(f"   最小差异: {min_diff_before:.8f}")
            print(f"   平均差异: {avg_diff_before:.8f}")
            
            # 判断差异是否足够大
            if max_diff_before < 1e-6:
                print(f"   ⚠️  严重警告: 聚合前所有agent的参数已经几乎相同（差异 < 1e-6）！")
                print(f"      这意味着：1) 初始化时参数相同 2) 之前的聚合已经使参数相同 3) 训练没有产生差异")
                print(f"      如果聚合前参数已相同，那么无论使用什么FL配置，结果都会相同！")
                print(f"      建议：使用 --use_different_seeds_per_agent 开关")
            elif max_diff_before < 0.01:
                print(f"   ⚠️  警告: 聚合前参数差异很小（< 0.01），FL配置的影响可能不明显")
                print(f"      建议：1) 使用 --use_different_seeds_per_agent 开关")
                print(f"           2) 降低聚合频率（增加 target_average_step）")
                print(f"           3) 使用软聚合保留更多本地参数")
            elif max_diff_before < 0.05:
                print(f"   ✅ 聚合前参数有差异（{max_diff_before:.4f}），但差异较小")
                print(f"      建议：使用软聚合或降低聚合频率以保持参数多样性")
            else:
                print(f"   ✅ 聚合前参数差异足够大（{max_diff_before:.4f}），FL配置应该能生效")
        
        # 验证：记录聚合前的参数范围，用于调试
        param_ranges_before = {}
        for i in range(self.n_veh):
            w_1_sample = self.sesses[i].run(self.w_1)
            param_ranges_before[i] = {
                'w_1_min': np.min(w_1_sample),
                'w_1_max': np.max(w_1_sample),
                'w_1_mean': np.mean(w_1_sample)
            }
        
        for i in range(self.n_veh):
            weight = weights[i]
            # 特征提取层 (Encoder) - 始终聚合
            w_1_mean += self.sesses[i].run(self.w_1) * weight
            w_2_mean += self.sesses[i].run(self.w_2) * weight
            b_1_mean += self.sesses[i].run(self.b_1) * weight
            b_2_mean += self.sesses[i].run(self.b_2) * weight
            
            # 决策层 (Task-specific Heads) - 只有在非分层模式下才聚合
            if not layer_wise:
                w_3_mean += self.sesses[i].run(self.w_3) * weight
                w_power_mu_mean += self.sesses[i].run(self.w_power_mu) * weight
                w_power_sigma_mean += self.sesses[i].run(self.w_power_sigma) * weight
                w_RB_mean += self.sesses[i].run(self.w_RB) * weight
                w_rho_mu_mean += self.sesses[i].run(self.w_rho_mu) * weight
                w_rho_sigma_mean += self.sesses[i].run(self.w_rho_sigma) * weight
                w_v_mean += self.sesses[i].run(self.w_v) * weight

                b_3_mean += self.sesses[i].run(self.b_3) * weight
                b_power_mu_mean += self.sesses[i].run(self.b_power_mu) * weight
                b_power_sigma_mean += self.sesses[i].run(self.b_power_sigma) * weight
                b_RB_mean += self.sesses[i].run(self.b_RB) * weight
                b_rho_mu_mean += self.sesses[i].run(self.b_rho_mu) * weight
                b_rho_sigma_mean += self.sesses[i].run(self.b_rho_sigma) * weight
                b_v_mean += self.sesses[i].run(self.b_v) * weight
        
        # 验证：检查聚合后的参数范围
        print(f"📊 聚合验证: w_1_mean范围=[{np.min(w_1_mean):.4f}, {np.max(w_1_mean):.4f}], 均值={np.mean(w_1_mean):.4f}")
        if layer_wise:
            print(f"📊 分层聚合: 只更新特征层(w_1,w_2,b_1,b_2)，决策层保持不变")
        else:
            print(f"📊 标准聚合: 更新所有层")

        # 软聚合：混合聚合参数和本地参数
        for i in range(self.n_veh):
            if aggregation_weight < 1.0:
                # 软聚合：保留部分本地参数
                # 获取当前本地参数 - 特征提取层
                old_w_1 = self.sesses[i].run(self.w_1)
                old_w_2 = self.sesses[i].run(self.w_2)
                old_b_1 = self.sesses[i].run(self.b_1)
                old_b_2 = self.sesses[i].run(self.b_2)
                
                # 获取决策层参数（仅非分层模式需要）
                if not layer_wise:
                    old_w_3 = self.sesses[i].run(self.w_3)
                    old_w_power_mu = self.sesses[i].run(self.w_power_mu)
                    old_w_power_sigma = self.sesses[i].run(self.w_power_sigma)
                    old_w_RB = self.sesses[i].run(self.w_RB)
                    old_w_rho_mu = self.sesses[i].run(self.w_rho_mu)
                    old_w_rho_sigma = self.sesses[i].run(self.w_rho_sigma)
                    old_w_v = self.sesses[i].run(self.w_v)
                    
                    old_b_3 = self.sesses[i].run(self.b_3)
                    old_b_power_mu = self.sesses[i].run(self.b_power_mu)
                    old_b_power_sigma = self.sesses[i].run(self.b_power_sigma)
                    old_b_RB = self.sesses[i].run(self.b_RB)
                    old_b_rho_mu = self.sesses[i].run(self.b_rho_mu)
                    old_b_rho_sigma = self.sesses[i].run(self.b_rho_sigma)
                    old_b_v = self.sesses[i].run(self.b_v)
                
                # 软聚合：混合新旧参数 - 特征提取层
                # new_param = aggregation_weight * aggregated_param + (1 - aggregation_weight) * local_param
                new_w_1 = aggregation_weight * w_1_mean + (1 - aggregation_weight) * old_w_1
                new_w_2 = aggregation_weight * w_2_mean + (1 - aggregation_weight) * old_w_2
                new_b_1 = aggregation_weight * b_1_mean + (1 - aggregation_weight) * old_b_1
                new_b_2 = aggregation_weight * b_2_mean + (1 - aggregation_weight) * old_b_2
                
                # 验证：检查软聚合是否生效（第一个智能体）
                if i == 0:
                    change_w_1 = np.mean(np.abs(new_w_1 - old_w_1))
                    change_agg = np.mean(np.abs(w_1_mean - old_w_1))
                    print(f"💡 软聚合验证 (Agent 0): w_1变化={change_w_1:.6f}, 硬替换变化={change_agg:.6f}, 软聚合比例={aggregation_weight:.2f}")
                
                self.sesses[i].run(self.w_1.assign(new_w_1))
                self.sesses[i].run(self.w_2.assign(new_w_2))
                self.sesses[i].run(self.b_1.assign(new_b_1))
                self.sesses[i].run(self.b_2.assign(new_b_2))
                
                # 决策层：只有在非分层模式下才进行聚合更新
                if not layer_wise:
                    self.sesses[i].run(self.w_3.assign(aggregation_weight * w_3_mean + (1 - aggregation_weight) * old_w_3))
                    self.sesses[i].run(self.w_power_mu.assign(aggregation_weight * w_power_mu_mean + (1 - aggregation_weight) * old_w_power_mu))
                    self.sesses[i].run(self.w_power_sigma.assign(aggregation_weight * w_power_sigma_mean + (1 - aggregation_weight) * old_w_power_sigma))
                    self.sesses[i].run(self.w_RB.assign(aggregation_weight * w_RB_mean + (1 - aggregation_weight) * old_w_RB))
                    self.sesses[i].run(self.w_rho_mu.assign(aggregation_weight * w_rho_mu_mean + (1 - aggregation_weight) * old_w_rho_mu))
                    self.sesses[i].run(self.w_rho_sigma.assign(aggregation_weight * w_rho_sigma_mean + (1 - aggregation_weight) * old_w_rho_sigma))
                    self.sesses[i].run(self.w_v.assign(aggregation_weight * w_v_mean + (1 - aggregation_weight) * old_w_v))
                    
                    self.sesses[i].run(self.b_3.assign(aggregation_weight * b_3_mean + (1 - aggregation_weight) * old_b_3))
                    self.sesses[i].run(self.b_power_mu.assign(aggregation_weight * b_power_mu_mean + (1 - aggregation_weight) * old_b_power_mu))
                    self.sesses[i].run(self.b_power_sigma.assign(aggregation_weight * b_power_sigma_mean + (1 - aggregation_weight) * old_b_power_sigma))
                    self.sesses[i].run(self.b_RB.assign(aggregation_weight * b_RB_mean + (1 - aggregation_weight) * old_b_RB))
                    self.sesses[i].run(self.b_rho_mu.assign(aggregation_weight * b_rho_mu_mean + (1 - aggregation_weight) * old_b_rho_mu))
                    self.sesses[i].run(self.b_rho_sigma.assign(aggregation_weight * b_rho_sigma_mean + (1 - aggregation_weight) * old_b_rho_sigma))
                    self.sesses[i].run(self.b_v.assign(aggregation_weight * b_v_mean + (1 - aggregation_weight) * old_b_v))
            else:
                # 硬替换（原有逻辑）：完全使用聚合参数
                # 特征提取层 - 始终更新
                self.sesses[i].run(self.w_1.assign(w_1_mean))
                self.sesses[i].run(self.w_2.assign(w_2_mean))
                self.sesses[i].run(self.b_1.assign(b_1_mean))
                self.sesses[i].run(self.b_2.assign(b_2_mean))
                
                # 决策层 - 只有在非分层模式下才更新
                if not layer_wise:
                    self.sesses[i].run(self.w_3.assign(w_3_mean))
                    self.sesses[i].run(self.w_power_mu.assign(w_power_mu_mean))
                    self.sesses[i].run(self.w_power_sigma.assign(w_power_sigma_mean))
                    self.sesses[i].run(self.w_RB.assign(w_RB_mean))
                    self.sesses[i].run(self.w_rho_mu.assign(w_rho_mu_mean))
                    self.sesses[i].run(self.w_rho_sigma.assign(w_rho_sigma_mean))
                    self.sesses[i].run(self.w_v.assign(w_v_mean))
                    
                    self.sesses[i].run(self.b_3.assign(b_3_mean))
                    self.sesses[i].run(self.b_power_mu.assign(b_power_mu_mean))
                    self.sesses[i].run(self.b_power_sigma.assign(b_power_sigma_mean))
                    self.sesses[i].run(self.b_RB.assign(b_RB_mean))
                    self.sesses[i].run(self.b_rho_mu.assign(b_rho_mu_mean))
                    self.sesses[i].run(self.b_rho_sigma.assign(b_rho_sigma_mean))
                    self.sesses[i].run(self.b_v.assign(b_v_mean))
        
        # 🔍 关键验证：检查聚合后参数是否真的被更新
        # 注意：聚合后参数相同是正常的（硬替换），关键是要看聚合前的差异
        if len(self.sesses) >= 2:
            # 比较第一个和第二个agent的参数（聚合后）
            w_1_agent0_after = self.sesses[0].run(self.w_1)
            w_1_agent1_after = self.sesses[1].run(self.w_1)
            param_diff_after = np.mean(np.abs(w_1_agent0_after - w_1_agent1_after))
            
            # 检查聚合后的参数与聚合前的差异
            if w_1_before_agg is not None:
                # 检查聚合后的参数是否真的改变了
                agg_change_agent0 = np.mean(np.abs(w_1_agent0_after - w_1_before_agg))
                # 检查聚合均值与聚合前第一个agent的差异
                agg_change_mean = np.mean(np.abs(w_1_mean - w_1_before_agg))
                
                print(f"🔍 聚合后验证（聚合后参数相同是正常的）:")
                print(f"   Agent0 vs Agent1 参数差异（聚合后）: {param_diff_after:.8f} {'✅ 正常（硬替换）' if param_diff_after < 1e-6 else '⚠️  异常'}")
                print(f"   Agent0 聚合前后变化: {agg_change_agent0:.8f}")
                print(f"   聚合均值 vs Agent0聚合前: {agg_change_mean:.8f}")
                
                # 聚合后参数相同是正常的，不需要警告
                if param_diff_after > 1e-6:
                    print(f"   ⚠️  警告: 聚合后参数应该相同（硬替换），但检测到差异！")
                    print(f"      这可能意味着：参数更新失败或使用了软聚合")
                
                if agg_change_agent0 < 1e-6:
                    print(f"   ⚠️  警告: Agent0 聚合前后参数几乎没有变化（变化 < 1e-6）！")
                    print(f"      这可能意味着：1) 聚合权重=0 2) 聚合均值与本地参数相同 3) 参数更新失败")
                if agg_change_mean < 1e-6:
                    print(f"   ⚠️  警告: 聚合均值与聚合前参数几乎相同（差异 < 1e-6）！")
                    print(f"      这可能意味着：所有agent的参数在聚合前就已经相同")
            else:
                print(f"🔍 聚合后验证: Agent0 vs Agent1 参数差异（聚合后）={param_diff_after:.8f}")