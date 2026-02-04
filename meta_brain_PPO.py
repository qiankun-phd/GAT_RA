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
        self.Loss_value = -L
        power_sample = tf.squeeze(power_dist.sample(1), axis=0)
        rho_sample = tf.squeeze(rho_distribution.sample(1), axis=0)
        self.choose_action_op = tf.concat([
            tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)),
            power_sample,
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

            saver = tf.train.Saver()
            v = tf.nn.relu(tf.add(tf.matmul(layer_3_b, self.w_v), self.b_v), name='v_layer')

            power_mu, power_sigma = power_mu, power_sigma + sigma_add
            rho_mu, rho_sigma = rho_mu, rho_sigma + sigma_add
            power_distribution = tf.distributions.Normal(loc=power_mu, scale=power_sigma)
            rho_distribution = tf.distributions.Normal(loc=rho_mu, scale=rho_sigma)
        params = tf.global_variables(scope)
        return power_distribution, RB_distribution, rho_distribution, v, params, saver

    def get_v(self, s, sess):
        return sess.run(self.v, {
            self.s_input: np.array([s])
        }).squeeze()

    def choose_action(self, s, sess):
        a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        clipped_a[2] = np.clip(a[2], -self.a_bound[2], self.a_bound[2])
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