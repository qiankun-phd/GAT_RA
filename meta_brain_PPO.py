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


        pi, RB_distribution, rho_distribution, self.v, params, self.saver = self._build_net('network', True)
        old_pi, old_RB_distribution, old_rho_distribution, old_v, old_params, _ = self._build_net('old_network', False)
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.v_pred_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_pred_next')
        self.gae = tf.placeholder(dtype=tf.float32, shape=[None], name='gae')

        GAE_advantage = self.gae
        # GAE_advantage = get_gaes(self.discounted_r)

        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        RB_action = self.a[:,0]
        power_action = self.a[:,1]
        rho_action = self.a[:,2]  # Compression ratio
        
        # Power probability ratio
        ratio = pi.prob(power_action) / (old_pi.prob(power_action) + 1e-8)
        ratio = tf.clip_by_value(ratio, 1e-6, 1e6)
        
        # Rho probability ratio（使用Normal分布，需要clip到[0,1]范围）
        rho_action_clipped = tf.clip_by_value(rho_action, 0.0, 1.0)
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
        S = tf.reduce_mean(pi.entropy() + RB_distribution.entropy() + rho_distribution.entropy())

        # Total loss
        L = L_clip + L_RB + L_rho - c1 * L_vf + c2 * S
        self.Loss = [L_clip, L_RB, L_rho, L_vf, S]
        self.Loss_value = -L
        # 动作采样：RB (Categorical) + Power (Normal) + Rho (Normal，与power相同)
        # 对于rho，采样后需要clip到[0,1]范围（因为使用Normal分布）
        rho_sample = tf.squeeze(rho_distribution.sample(1), axis=0)
        rho_sample = tf.clip_by_value(rho_sample, 0.0, 1.0)  # 确保rho在[0,1]范围内
        self.choose_action_op = tf.concat([
            tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)), 
            tf.squeeze(pi.sample(1), axis=0),
            rho_sample  # 新增：压缩比
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
            self.w_mu = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_sigma = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_RB = tf.Variable(initializer(shape=(n_hidden_2, self.n_RB)), trainable=trainable)
            # 新增：rho的网络参数（Normal分布，与power相同）
            self.w_rho_mu = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_rho_sigma = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_v = tf.Variable(initializer(shape=(n_hidden_3, 1)), trainable=trainable)

            self.b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1), trainable=trainable)
            self.b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1), trainable=trainable)
            self.b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1), trainable=trainable)
            self.b_mu = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_sigma = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_RB = tf.Variable(tf.truncated_normal([self.n_RB], stddev=0.1), trainable=trainable)
            # 新增：rho的bias（Normal分布，与power相同）
            self.b_rho_mu = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_rho_sigma = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_v = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)

            layer_p1 = tf.nn.relu(tf.add(tf.matmul(self.s_input, self.w_1), self.b_1), name='p_1')
            layer_1_b = tf.layers.batch_normalization(layer_p1)
            layer_p2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, self.w_2), self.b_2), name='p_2')
            layer_2_b = tf.layers.batch_normalization(layer_p2)
            layer_p3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, self.w_3), self.b_3), name='p_3')
            layer_3_b = tf.layers.batch_normalization(layer_p3)

            mu = tf.nn.tanh(tf.add(tf.matmul(layer_2_b, self.w_mu), self.b_mu), name='mu_layer')
            sigma = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_sigma), self.b_sigma), name='sigma_layer')
            RB_probs = tf.nn.softmax(tf.add(tf.matmul(layer_2_b, self.w_RB), self.b_RB), name='RB_layer')
            RB_distribution = tf.distributions.Categorical(probs=RB_probs)
            
            # 新增：rho的Normal分布（与power相同的方式，压缩比 rho ∈ [0,1]）
            # 使用sigmoid将mu映射到[0,1]范围
            rho_mu = tf.nn.sigmoid(tf.add(tf.matmul(layer_2_b, self.w_rho_mu), self.b_rho_mu), name='rho_mu_layer')
            rho_sigma = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_sigma), self.b_rho_sigma), name='rho_sigma_layer')
            rho_mu, rho_sigma = rho_mu, rho_sigma + sigma_add
            rho_distribution = tf.distributions.Normal(loc=rho_mu, scale=rho_sigma)

            saver = tf.train.Saver()


            # layer_p1 = tf.layers.dense(self.s_input, 100, tf.nn.relu, name='p_0', trainable=trainable)
            # layer_p2 = tf.layers.dense(layer_p1, 100, tf.nn.relu, name='p_1', trainable=trainable)
            # layer_p3 = tf.layers.dense(layer_p2, 100, tf.nn.relu, name='p_2', trainable=trainable)
            # mu = tf.layers.dense(layer_p3, self.a_dim, tf.nn.tanh, name='mu_layer', trainable=trainable)
            # sigma = tf.layers.dense(layer_p3, self.a_dim, tf.nn.softplus, name='sigma_layer', trainable=trainable)

            # layer_r0 = tf.layers.dense(self.s_input, 500, activation=tf.nn.relu, name='r_0', trainable=trainable)
            # layer_r1 = tf.layers.dense(layer_r0, 250, activation=tf.nn.relu, name='r_1', trainable=trainable)
            # layer_r2 = tf.layers.dense(layer_r1, 120, activation=tf.nn.relu, name='r_2', trainable=trainable)
            # RB = tf.layers.dense(layer_r2, self.n_RB, activation=tf.nn.softmax, name='RB_layer', trainable=trainable)

            # 状态价值函数 v 与策略 π 共享同一套神经网络参数
            v =  tf.nn.relu(tf.add(tf.matmul(layer_3_b, self.w_v), self.b_v), name='v_layer')

            # mu, sigma = mu, sigma + sigma_add
            mu, sigma = mu, sigma + sigma_add

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.global_variables(scope)
        return norm_dist, RB_distribution, rho_distribution, v, params, saver

    def get_v(self, s, sess):
        return sess.run(self.v, {
            self.s_input: np.array([s])
        }).squeeze()

    def choose_action(self, s, sess):
        a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]  # RB (already integer from Categorical)
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])  # Power
        clipped_a[2] = np.clip(a[2], 0.0, 1.0)  # Rho (compression ratio) ∈ [0,1]
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