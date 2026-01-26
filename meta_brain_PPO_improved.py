import numpy as np
import os
import random
import copy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

class PPOImproved(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr_a, lr_c, K, n_RB, sess, task_id=None):
        self.a_bound = a_bound
        self.K = K
        self.s_dim = s_dim
        self.a_dim = 3  # RB_choice + Power + Compression Ratio (rho)
        self.s_input = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')
        self.n_RB = n_RB
        self.sess = sess
        self.gamma = args.gamma
        self.GAE_discount = args.lambda_advantage
        self.task_id = task_id
        
        # 学习率调度
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr_a_decay = tf.train.exponential_decay(
            learning_rate=lr_a,
            global_step=self.global_step,
            decay_steps=100,
            decay_rate=0.99,
            staircase=False,
            name='lr_a_decay'
        )
        self.lr_c_decay = tf.train.exponential_decay(
            learning_rate=lr_c,
            global_step=self.global_step,
            decay_steps=100,
            decay_rate=0.99,
            staircase=False,
            name='lr_c_decay'
        )

        # Placeholders（必须在构建网络前定义）
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.v_pred_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_pred_next')
        self.gae = tf.placeholder(dtype=tf.float32, shape=[None], name='gae')
        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        
        # 任务嵌入（如果提供task_id）
        if task_id is not None:
            self.task_embedding = tf.placeholder(tf.float32, shape=[None, 1], name='task_embedding')
        else:
            self.task_embedding = None

        # 构建网络和placeholders
        power_dist, RB_distribution, rho_distribution, self.v, params, self.saver = self._build_net('network', True)
        old_power_dist, old_RB_distribution, old_rho_distribution, old_v, old_params, _ = self._build_net('old_network', False)

        GAE_advantage = self.gae
        
        RB_action = self.a[:,0]
        power_action = self.a[:,1]
        rho_action = self.a[:,2]  # Compression ratio
        
        # Beta分布处理：power和rho都使用Beta分布，输出[0,1]
        # Power需要从[-bound, bound]映射到[0,1]来计算概率
        power_normalized = (power_action + self.a_bound[1]) / (2 * self.a_bound[1] + 1e-8)
        power_normalized = tf.clip_by_value(power_normalized, 1e-6, 1.0 - 1e-6)  # 避免边界值
        ratio = power_dist.prob(power_normalized) / (old_power_dist.prob(power_normalized) + 1e-8)
        ratio = tf.clip_by_value(ratio, 1e-6, 1e6)
        
        # Rho已经是[0,1]范围，直接使用Beta分布
        rho_action_clipped = tf.clip_by_value(rho_action, 1e-6, 1.0 - 1e-6)  # 避免边界值
        ratio_rho = rho_distribution.prob(rho_action_clipped) / (old_rho_distribution.prob(rho_action_clipped) + 1e-8)
        ratio_rho = tf.clip_by_value(ratio_rho, 1e-6, 1e6)

        L_vf = tf.reduce_mean(tf.square(self.reward + self.gamma * self.v_pred_next - self.v))
        
        # Power PPO loss
        L_clip = tf.reduce_mean(tf.minimum(
            ratio * GAE_advantage,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))

        # RB PPO loss
        ratio_RB = RB_distribution.prob(RB_action) / (old_RB_distribution.prob(RB_action) + 1e-8)
        ratio_RB = tf.clip_by_value(ratio_RB, 1e-6, 1e6)
        L_RB = tf.reduce_mean(tf.minimum(
            ratio_RB * GAE_advantage,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio_RB, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))
        
        # Rho PPO loss
        L_rho = tf.reduce_mean(tf.minimum(
            ratio_rho * GAE_advantage,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio_rho, 1 - epsilon, 1 + epsilon) * GAE_advantage
        ))

        # Entropy (for exploration)
        S = tf.reduce_mean(power_dist.entropy() + RB_distribution.entropy() + rho_distribution.entropy())

        # Total loss
        L = L_clip + L_RB + L_rho - c1 * L_vf + c2 * S
        self.Loss = [L_clip, L_RB, L_rho, L_vf, S]
        self.Loss_value = -L
        
        # 梯度裁剪
        optimizer = tf.train.AdamOptimizer(self.lr_a_decay)
        gradients, variables = zip(*optimizer.compute_gradients(-L))
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  # 梯度裁剪
        self.gradient_norm = gradient_norm  # 用于监控
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)
        
        # 动作采样：RB (Categorical) + Power (Beta) + Rho (Beta)
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
        
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        
        # TensorBoard summaries
        self.create_summaries()
        
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, scope, trainable):
        with tf.variable_scope(scope):
            initializer = tf.truncated_normal_initializer(stddev=0.1)

            # 如果有任务嵌入，扩展输入维度（新网络和旧网络都要保持一致）
            if self.task_embedding is not None:
                # 将任务嵌入与状态拼接
                extended_input = tf.concat([self.s_input, self.task_embedding], axis=1)
                input_dim = self.s_dim + 1
            else:
                extended_input = self.s_input
                input_dim = self.s_dim

            self.w_1 = tf.Variable(initializer(shape=(input_dim, n_hidden_1)), trainable=trainable)
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

            layer_p1 = tf.nn.relu(tf.add(tf.matmul(extended_input, self.w_1), self.b_1), name='p_1')
            layer_1_b = tf.layers.batch_normalization(layer_p1, training=trainable)
            layer_p2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, self.w_2), self.b_2), name='p_2')
            layer_2_b = tf.layers.batch_normalization(layer_p2, training=trainable)
            layer_p3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, self.w_3), self.b_3), name='p_3')
            layer_3_b = tf.layers.batch_normalization(layer_p3, training=trainable)

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

            saver = tf.train.Saver()

            # 状态价值函数 v 与策略 π 共享同一套神经网络参数
            v =  tf.nn.relu(tf.add(tf.matmul(layer_3_b, self.w_v), self.b_v), name='v_layer')
        params = tf.global_variables(scope)
        return power_distribution, RB_distribution, rho_distribution, v, params, saver

    def create_summaries(self):
        """创建TensorBoard summaries"""
        with tf.variable_scope('summaries'):
            # 损失监控
            self.loss_summary = []
            loss_names = ['L_clip', 'L_RB', 'L_rho', 'L_vf', 'entropy']
            for i, name in enumerate(loss_names):
                summary = tf.placeholder(tf.float32, name=f'{name}_ph')
                self.loss_summary.append(summary)
                tf.summary.scalar(f'Loss/{name}', summary)
            
            # 学习率监控
            tf.summary.scalar('Learning_Rate/lr_a', self.lr_a_decay)
            tf.summary.scalar('Learning_Rate/lr_c', self.lr_c_decay)
            
            # 梯度范数监控
            tf.summary.scalar('Gradient/gradient_norm', self.gradient_norm)
            
            # 奖励和价值监控
            self.reward_ph = tf.placeholder(tf.float32, name='reward_ph')
            self.value_ph = tf.placeholder(tf.float32, name='value_ph')
            tf.summary.scalar('Training/average_reward', self.reward_ph)
            tf.summary.scalar('Training/average_value', self.value_ph)
            
            self.merged_summary = tf.summary.merge_all()

    def get_v(self, s, sess, task_embedding=None):
        feed_dict = {self.s_input: np.array([s])}
        if self.task_embedding is not None and task_embedding is not None:
            feed_dict[self.task_embedding] = np.array([[task_embedding]])
        return sess.run(self.v, feed_dict).squeeze()

    def choose_action(self, s, sess, task_embedding=None):
        feed_dict = {self.s_input: s[np.newaxis, :]}
        if self.task_embedding is not None and task_embedding is not None:
            feed_dict[self.task_embedding] = np.array([[task_embedding]])
        
        a = np.squeeze(sess.run(self.choose_action_op, feed_dict))
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]  # RB (Categorical)
        # Power已经是Beta分布映射后的值，在[-bound, bound]范围内，但为了安全还是clip一下
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        # Rho是Beta分布输出，天然在[0,1]范围内，但为了安全还是clip一下
        clipped_a[2] = np.clip(a[2], 0.0, 1.0)
        return clipped_a

    def train(self, s, a, gae, reward, v_pred_next, sess, task_embedding=None, summary_writer=None, episode=None):
        """训练网络（单步更新）"""
        sess.run(self.update_params_op)
        
        # 构建feed_dict
        feed_dict = {
            self.s_input: s,
            self.a: a,
            self.reward: reward,
            self.v_pred_next: v_pred_next,
            self.gae: gae
        }
        if self.task_embedding is not None and task_embedding is not None:
            # task_embedding should be repeated for batch size
            batch_size = s.shape[0]
            task_emb_batch = np.full((batch_size, 1), task_embedding)
            feed_dict[self.task_embedding] = task_emb_batch
        
        # K epochs更新
        for i in range(self.K):
            sess.run(self.train_op, feed_dict)
        
        # 获取损失
        loss_values, gradient_norm_val = sess.run([self.Loss, self.gradient_norm], feed_dict)
        
        # TensorBoard日志
        if summary_writer is not None and episode is not None:
            # 创建summary feed_dict
            summary_feed_dict = feed_dict.copy()
            for i, loss_val in enumerate(loss_values):
                summary_feed_dict[self.loss_summary[i]] = loss_val
            summary_feed_dict[self.reward_ph] = np.mean(reward)
            # 计算价值时也需要提供任务嵌入
            v_feed_dict = {self.s_input: s}
            if self.task_embedding is not None and task_embedding is not None:
                batch_size = s.shape[0]
                task_emb_batch = np.full((batch_size, 1), task_embedding)
                v_feed_dict[self.task_embedding] = task_emb_batch
            summary_feed_dict[self.value_ph] = np.mean(sess.run(self.v, v_feed_dict))
            
            summary = sess.run(self.merged_summary, summary_feed_dict)
            summary_writer.add_summary(summary, episode)
        
        return loss_values, gradient_norm_val

    def train_multi_step(self, s, a, gae, reward, v_pred_next, sess, inner_steps=3, task_embedding=None, summary_writer=None, episode=None):
        """多步适应训练（MAML风格的内循环）"""
        # 保存原始参数
        original_params = sess.run(tf.global_variables('network'))
        
        sess.run(self.update_params_op)
        
        # 构建feed_dict
        feed_dict = {
            self.s_input: s,
            self.a: a,
            self.reward: reward,
            self.v_pred_next: v_pred_next,
            self.gae: gae
        }
        if self.task_embedding is not None and task_embedding is not None:
            batch_size = s.shape[0]
            task_emb_batch = np.full((batch_size, 1), task_embedding)
            feed_dict[self.task_embedding] = task_emb_batch
        
        # 内循环：多步适应
        losses_history = []
        for step in range(inner_steps):
            # K epochs更新
            for i in range(self.K):
                sess.run(self.train_op, feed_dict)
            
            # 记录损失
            loss_values, gradient_norm_val = sess.run([self.Loss, self.gradient_norm], feed_dict)
            losses_history.append(loss_values)
        
        # TensorBoard日志（记录最后一步的损失）
        if summary_writer is not None and episode is not None:
            summary_feed_dict = feed_dict.copy()
            for i, loss_val in enumerate(loss_values):
                summary_feed_dict[self.loss_summary[i]] = loss_val
            summary_feed_dict[self.reward_ph] = np.mean(reward)
            # 计算价值时也需要提供任务嵌入
            v_feed_dict = {self.s_input: s}
            if self.task_embedding is not None and task_embedding is not None:
                batch_size = s.shape[0]
                task_emb_batch = np.full((batch_size, 1), task_embedding)
                v_feed_dict[self.task_embedding] = task_emb_batch
            summary_feed_dict[self.value_ph] = np.mean(sess.run(self.v, v_feed_dict))
            
            summary = sess.run(self.merged_summary, summary_feed_dict)
            summary_writer.add_summary(summary, episode)
        
        return losses_history, gradient_norm_val

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE calculation with improved numerical stability
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Flatten if needed
        if len(rewards.shape) > 1:
            original_shape = rewards.shape
            rewards = rewards.flatten()
            v_preds = v_preds.flatten()
            v_preds_next = v_preds_next.flatten()
        else:
            original_shape = None
        
        deltas = rewards + self.gamma * v_preds_next - v_preds

        # 计算GAE(lambda = 1), 参见 ppo paper eq(11)
        gaes = copy.deepcopy(deltas)

        # 倒序计算GAE
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
        
        # Reshape back if needed
        if original_shape is not None:
            gaes = np.array(gaes).reshape(original_shape)
        
        return gaes

    def normalize_advantages(self, advantages, method='standard'):
        """
        Normalize advantages with different methods
        """
        advantages = np.array(advantages)
        
        if method == 'standard':
            # 标准归一化
            mean = np.mean(advantages)
            std = np.std(advantages)
            if std > 1e-8:
                return (advantages - mean) / (std + 1e-8)
            else:
                return advantages
        
        elif method == 'robust':
            # 鲁棒归一化（使用分位数）
            q25, q75 = np.percentile(advantages, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-8:
                median = np.median(advantages)
                return (advantages - median) / (iqr + 1e-8)
            else:
                return advantages
        
        elif method == 'none':
            # 不归一化
            return advantages
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def averaging_model(self, n_veh):
        """
        NOTE: This method should NOT be called in meta training.
        Meta training uses a single agent (self.sess), not multiple agents (self.sesses).
        This method is kept for interface compatibility but will raise an error if called.
        Federated learning averaging is handled in PPO_brain_AC.py, not here.
        """
        raise NotImplementedError(
            "averaging_model should not be called in meta training. "
            "Meta training uses a single agent (self.sess), not multiple agents (self.sesses). "
            "This method is only for federated learning in main_PPO_AC.py."
        )