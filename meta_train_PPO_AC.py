import random
import concurrent.futures
import numpy as np
import os
# import gym
from meta_brain_PPO import PPO
import matplotlib.pyplot as plt
from Environment_marl_indoor import Environ as Environment_marl_general
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from arguments import get_args
args = get_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.set_random_seed)

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth = True


# env = gym.make('Pendulum-v1')
# env = env.unwrapped
meta_episode = args.meta_episode

IS_PPO = args.IS_PPO if hasattr(args, 'IS_PPO') else True  # 默认使用PPO
n_veh = args.n_veh_list
n_RB = args.n_RB

sigma_add = args.sigma_add

BATCH_SIZE = args.meta_batch_size
i_episode = 0
ACTOR_NUM = 1
current_fed_times = 0

def get_state(env, idx=(0, 0), n_veh = 1, ind_episode=0.):
    """ Get state from the environment """
    cellular_fast = (env.cellular_channels_with_fastfading[idx[0], :] - env.cellular_channels_abs[idx[0]] + 10) / 35
    cellular_abs = (env.cellular_channels_abs[idx[0]] - 80) / 60.0
    success = env.success[idx[0]]
    channel_choice = env.channel_choice / n_veh
    vehicle_vector = np.zeros(n_RB)
    for i in range (n_veh):
        vehicle_vector[i] = 1 / n_veh
    return np.concatenate((np.reshape(cellular_fast, -1), np.reshape(cellular_abs, -1), np.reshape(channel_choice, -1), vehicle_vector,
                           np.asarray([success, ind_episode / (meta_episode)])))

def save_models(sess, model_path, saver):
    """ Save models to the current directory with the name filename """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)

ppoes = []
envs = []
sess = tf.Session(config=my_config)

# 固定使用SEE优化目标（只优化语义能量效率）
optimization_target = 'SEE'
beta = args.beta if hasattr(args, 'beta') else 0.5
circuit_power = args.circuit_power if hasattr(args, 'circuit_power') else 0.06

for k in range(len(n_veh)):
    env = Environment_marl_general(
        n_veh[k], n_RB, 
        beta=beta, 
        circuit_power=circuit_power, 
        optimization_target=optimization_target,
        semantic_A1=1.0,
        semantic_A2=0.2,
        semantic_C1=5.0,
        semantic_C2=2.0
    )
    env.new_random_game()
    envs.append(env)

T_TIMESTEPS = int(env.time_slow / (env.time_fast))
state_dim = len(get_state(env=env))
action_dim = 3  # RB_choice + power + rho (compression ratio)
action_bound = []
action_bound.append(n_RB)  # RB bound
action_bound.append(args.RB_action_bound)  # Power bound
# 注意：rho使用Beta分布，范围固定为[0,1]，不需要action_bound

ppo = PPO(state_dim, action_bound, args.weight_for_L_vf, args.weight_for_entropy, args.epsilon, args.lr_meta_a, args.lr_meta_c, args.minibatch_steps, n_RB, sess)

executor = concurrent.futures.ThreadPoolExecutor(ACTOR_NUM)

def simulate():
    trans_all_user_task = []
    r_sum_task = []

    for k in range(len(n_veh)):
        envs[k].renew_positions()  # update vehicle position
        # env.renew_neighbor()
        envs[k].renew_BS_channel()  # update channel slow fading
        envs[k].renew_BS_channels_fastfading()
        state_alls = []
        action_alls = []
        v_pred_alls = []
        rewards = []
        state_all = []

        for i in range(n_veh[k]):
            state_all.append(get_state(envs[k], [i, 0], n_veh[k], i_episode))
        r_sum = 0
        trans_all_user = []

        for step in range(T_TIMESTEPS):
            action_all = []
            v_pred_all = []
            reward_all = []
            action_all_training = np.zeros([n_veh[k], 3], dtype='float32')

            for i in range(n_veh[k]):
                action = ppo.choose_action(state_all[i], sess)
                v_pred = ppo.get_v(state_all[i], sess).tolist()
                action_all.append(action)

                channel_action = action[0]
                amp = envs[k].cellular_power_dB_List[0] / (2 * action_bound[1])
                power_action = (action[1] + action_bound[1]) * amp
                # 压缩比rho：从action的第3维获取（PPO网络已输出）
                rho_action = action[2]  # 已经是[0,1]范围内的值（Beta分布输出）
                # 确保rho在[0, 1]范围内（双重保险）
                rho_action = np.clip(rho_action, 0.0, 1.0)
                
                action_all_training[i, 0] = channel_action
                action_all_training[i, 1] = power_action
                action_all_training[i, 2] = rho_action
                
                v_pred_all.append(v_pred)
            action_temp = action_all_training.copy()
            train_reward = envs[k].act_for_meta_training(action_temp, IS_PPO)
            for i in range(n_veh[k]):
                reward_all.append(train_reward)
                state_ = get_state(envs[k], [i, 0], n_veh[k], i_episode)
                state_all[i] = state_

            r_sum += train_reward

            state_alls = np.append(state_alls, np.asarray(state_all))
            action_alls = np.append(action_alls, np.asarray(action_all))
            v_pred_alls = np.append(v_pred_alls, np.asarray(v_pred_all))
            rewards = np.append(rewards, np.asarray(reward_all))
            v_preds_next = v_pred_alls

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        v_pred_alls = v_pred_alls.reshape([-1, n_veh[k]])
        v_preds_next = np.append(v_pred_alls[1:], np.zeros([n_veh[k]]))
        rewards = rewards.reshape([-1, n_veh[k]])
        v_pred_alls = v_pred_alls.reshape([-1, n_veh[k]])
        v_preds_next = v_preds_next.reshape([-1, n_veh[k]])

        gaes = ppo.get_gaes(rewards=rewards, v_preds=v_pred_alls, v_preds_next=v_preds_next)
        gaes = np.array(gaes).astype(dtype=np.float32)
        # 归一化GAE，添加epsilon防止除以0（当所有GAE值相同时）
        gaes_mean = gaes.mean()
        gaes_std = gaes.std()
        if gaes_std > 1e-8:
            gaes = (gaes - gaes_mean) / gaes_std
        # 如果所有GAE值相同（std=0），保持原值不变（不归一化）

        state_alls = np.reshape(state_alls, newshape=(-1, n_veh[k], state_dim))
        action_alls = np.reshape(action_alls, newshape=(-1, n_veh[k], action_dim))
        # gaes = np.reshape(gaes, newshape=(-1, n_veh[k], 1))
        # rewards = np.reshape(rewards, newshape=(-1, n_veh[k], 1))
        # v_preds_next = np.reshape(v_preds_next, newshape=(-1, n_veh[k], 1))

        trans_all_user = [state_alls, action_alls, gaes, rewards, v_preds_next]

        r_sum_task.append(r_sum / T_TIMESTEPS)
        trans_all_user_task.append(trans_all_user)

    return r_sum_task, trans_all_user_task

record_reward = []
loss_episode = []
for i in range(meta_episode):
    i_episode = i_episode + 1
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)

    r_avgs = []
    for f in futures:
        r_avg_task, trans_all_user_task = f.result()
        for k in range (len(n_veh)):
            r_avgs.append(r_avg_task[k])
        record_reward.append(np.sum(r_avgs))

    # record_reward.append(r_avgs)
    print('Episode:', i, 'Average Reward', r_avgs, 'Sum Reward', np.sum(r_avgs))
    # generalization
    loss_batch = []
    batch = []
    sampled_inp = []
    loss_all = []
    sample_indices = np.random.randint(low=0, high=trans_all_user_task[0][0].shape[0], size=BATCH_SIZE)

    for k in range (len(n_veh)):
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in trans_all_user_task[k]]  # sample training data
        for i in range(n_veh[k]):
            s = sampled_inp[0][:, i, :]
            a = sampled_inp[1][:, i, :]
            gae = sampled_inp[2][:, i]
            reward = sampled_inp[3][:, i]
            v_pred_next = sampled_inp[4][:, i]

            loss = ppo.train(s, a, gae, reward, v_pred_next, sess)
            # print('Task_'+ '%d:' % k +'loss_' + '%d:' % i, loss[0])
            # print('Entropy_'+'%d:' %i, entropy)
            loss_all.append(loss[1])
        loss_batch.append(sum(loss_all))
    loss_episode.append(sum(loss_batch) / BATCH_SIZE)
    print('Loss_episode: ', loss_episode[-1])

# 构建模型路径，包含优化目标信息（固定为SEE）
opt_target_str = optimization_target  # SEE
opt_suffix = opt_target_str  # SEE（不再需要beta参数）

model_path = args.save_path + 'AC_' + opt_suffix + '_' + '%s_' %sigma_add + '%d_' % meta_episode +'%s_' %args.lr_meta_a
save_models(sess, model_path, ppo.saver)

np.savetxt('./Train_data/Meta_Reward_'+ opt_suffix + '_' + '%s_' %sigma_add + '%d_' %meta_episode +'%s_' %args.lr_meta_a, record_reward)

# np.savetxt('loss_analysis', loss_eposide)
# plt.rcParams['figure.dpi'] = 300
# plt.figure(1)
# plt.grid()
# plt.plot(record_reward)
# plt.ylabel('Training Reward')
# plt.xlabel('Episodes')
# plt.legend()
# plt.show()
