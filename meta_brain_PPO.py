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

class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr_a, lr_c, K, n_RB, sess):
        self.a_bound = a_bound
        self.K = K
        self.s_dim = s_dim
        self.a_dim = 3  # RB_choice + Power + Compression Ratio (rho)
        self.s_input = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')
        self.n_RB = n_RB
        self.sess = sess
        self.gamma = args.gamma
        self.GAE_discount = args.lambda_advantage

        # with tf.variable_scope('critic'):
        #     l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        #     self.v = tf.layers.dense(l1, 1)
        #     self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        #     self.advantage = self.tfdc_r - self.v
        #     self.closs = tf.reduce_mean(tf.square(self.advantage))
        #     self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)


        power_dist, RB_distribution, rho_distribution, self.v, params, self.saver = self._build_net('network', True)
        old_power_dist, old_RB_distribution, old_rho_distribution, old_v, old_params, _ = self._build_net('old_network', False)
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.v_pred_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_pred_next')
        self.gae = tf.placeholder(dtype=tf.float32, shape=[None], name='gae')

        GAE_advantage = self.gae
        # GAE_advantage = get_gaes(self.discounted_r)

        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
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
        self.train_op = tf.train.AdamOptimizer(lr_a).minimize(-L)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, scope, trainable):

        with tf.variable_scope(scope):
            initializer = tf.compat.v1.keras.initializers.he_normal()

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

    def get_v(self, s, sess):
        return sess.run(self.v, {
            self.s_input: np.array([s])
        }).squeeze()

    def choose_action(self, s, sess):
        a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]  # RB (Categorical)
        # Power已经是Beta分布映射后的值，在[-bound, bound]范围内，但为了安全还是clip一下
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        # Rho是Beta分布输出，天然在[0,1]范围内，但为了安全还是clip一下
        clipped_a[2] = np.clip(a[2], 0.0, 1.0)
        return clipped_a

    def train(self, s, a, gae, reward, v_pred_next, sess):
        sess.run(self.update_params_op)
        # K epochs
        for i in range(self.K):
            sess.run(self.train_op, {
                self.s_input: s,
                self.a: a,
                self.reward: reward,
                self.v_pred_next: v_pred_next,
                self.gae: gae
            })
        loss = sess.run([self.Loss, self.Loss_value], {
                self.s_input: s,
                self.a: a,
                self.reward: reward,
                self.v_pred_next: v_pred_next,
                self.gae: gae
            })
        # entropy = sess.run(self.total_entropy, {
        #     self.s_input: s,
        #     self.a: a,
        #     self.discounted_r: discounted_r
        # })
        return loss

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t)
        :param v_preds: v(st)
        :param v_preds_next: v(st+1)
        :return:
        """
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]

        # 计算GAE(lambda = 1), 参见 ppo paper eq(11)
        gaes = copy.deepcopy(deltas)

        # 倒序计算GAE
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
        return gaes

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