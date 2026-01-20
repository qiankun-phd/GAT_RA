import random
import concurrent.futures
import numpy as np
# import gym
import os
from PPO_brain_AC import PPO
import matplotlib.pyplot as plt
import Environment_marl_indoor
import Environment_marl_urban_micro
import Environment_marl_urban_macro
import Environment_marl_rural_macro

from arguments import get_args
args = get_args()
import random
from tensorflow.compat.v1 import set_random_seed

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
set_random_seed(args.set_random_seed)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# env = gym.make('Pendulum-v1')
# env = env.unwrapped
meta_episode = args.meta_episode
target_average_step = args.target_average_step


IS_PPO = args.IS_PPO
IS_meta = args.Do_meta
n_veh = args.n_veh
n_neighbor = args.n_neighbor
n_RB = args.n_RB
n_BS = args.n_BS

env_indoor = Environment_marl_indoor.Environ(n_veh, n_neighbor, n_RB, n_BS)
env_cannon = Environment_marl_urban_micro.Environ(n_veh, n_neighbor, n_RB, n_BS)
env_urban = Environment_marl_urban_macro.Environ(n_veh, n_neighbor, n_RB, n_BS)
env_rural = Environment_marl_rural_macro.Environ(n_veh, n_neighbor, n_RB, n_BS)

env_choice = args.env_choice
if env_choice == 0:
    env = env_indoor
    env_label = "indoor"
elif env_choice ==1:
    env = env_cannon
    env_label = "cannon"
elif env_choice == 2:
    env = env_urban
    env_label = "urban"
else:
    env = env_rural
    env_label = "rural"
env.new_random_game()

GAMMA = args.gamma
BATCH_SIZE = args.meta_batch_size
i_episode = 0
n_episode = args.n_episode
ACTOR_NUM = 1
T_TIMESTEPS = int(env.time_slow / (env.time_fast))
current_fed_times = 0


def get_state(env, idx=(0, 0), n_veh = 0, ind_episode=0.):
    """ Get state from the environment """
    cellular_fast = (env.cellular_channels_with_fastfading[idx[0], :] - env.cellular_channels_abs[idx[0]] + 10) / 35
    cellular_abs = (env.cellular_channels_abs[idx[0]] - 80) / 60.0
    success = env.success[idx[0]]
    channel_choice = env.channel_choice / n_veh
    vehicle_vector = np.zeros(n_RB)
    for i in range (n_veh):
        vehicle_vector[i] = 1 / n_veh
    return np.concatenate((np.reshape(cellular_fast, -1), np.reshape(cellular_abs, -1), np.reshape(channel_choice, -1), vehicle_vector,
                           np.asarray([success, ind_episode / (n_episode)])))

def save_models(sess, model_path, saver):
    """ Save models to the current directory with the name filename """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)

state_dim = len(get_state(env=env))
action_dim = 2 # RB_choice + power
action_bound = []
action_bound.append(n_RB)
action_bound.append(args.RB_action_bound)

ppoes = []
ppoes = PPO(state_dim, action_bound, args.weight_for_L_vf, args.weight_for_entropy, args.epsilon, args.lr_main, args.lr_meta_a, args.minibatch_steps, n_veh, n_RB, IS_meta, meta_episode)


executor = concurrent.futures.ThreadPoolExecutor(ACTOR_NUM)


def simulate():
    env.renew_positions()  # update vehicle position
    # env.renew_neighbor()
    env.renew_BS_channel()  # update channel slow fading
    env.renew_BS_channels_fastfading()  # update channel fast fading
    r_sum = 0
    trans_all_user = []
    success_alls = []
    state_alls = []
    action_alls = []
    v_pred_alls = []
    rewards = []

    state_all = []
    for i in range(n_veh):
        for j in range(n_neighbor):
            state_all.append(get_state(env, [i, j], n_veh, i_episode))
    for step in range(T_TIMESTEPS):
        env.renew_BS_channels_fastfading()
        action_all = []
        v_pred_all = []
        reward_all = []
        action_all_training = np.zeros([n_veh, n_neighbor, 3], dtype='float32')

        for i in range(n_veh):
            for j in range(n_neighbor):
                action = ppoes.choose_action(state_all[i], ppoes.sesses[i * n_neighbor + j])
                v_pred = ppoes.get_v(state_all[i], ppoes.sesses[i * n_neighbor + j]).tolist()
                action_all.append(action)
                v_pred_all.append(v_pred)
                amp = env.cellular_power_dB_List[0] / (2 * action_bound[-1])
                power_action = (action[-1] + action_bound[-1]) * amp
                action_all_training[i, j, 0] = action[0]
                action_all_training[i, j, 1] = power_action
        # print("Channel selection: ", action_all_training[:, :, 0], "Power: ", action_all_training[:, : , 1])
        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp, IS_PPO)
        for i in range(n_veh):
            for j in range(n_neighbor):
                reward_all.append(train_reward)
                state_ = get_state(env, [i, j], n_veh, i_episode)
                state_all[i] = state_
        success_all = env.success
        r_sum += train_reward

        state_alls = np.append(state_alls, np.asarray(state_all))
        action_alls = np.append(action_alls, np.asarray(action_all))
        v_pred_alls = np.append(v_pred_alls, np.asarray(v_pred_all))
        rewards = np.append(rewards, np.asarray(reward_all))
        success_alls = np.append(success_alls, np.asarray(success_all))
        v_preds_next = v_pred_alls

    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = np.append(v_pred_alls[1:], np.zeros([n_veh]))
    rewards = rewards.reshape([-1, n_veh])
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = v_preds_next.reshape([-1, n_veh])
    success_alls = success_alls.reshape([-1, n_veh])

    gaes = ppoes.get_gaes(rewards=rewards, v_preds=v_pred_alls, v_preds_next=v_preds_next)

    state_alls = np.reshape(state_alls, newshape=(-1, n_veh, state_dim))
    action_alls = np.reshape(action_alls, newshape=(-1, n_veh, action_dim))
    gaes = np.array(gaes).astype(dtype=np.float32)
    gaes = (gaes - gaes.mean()) / (gaes.std())

    trans_all_user = [state_alls, action_alls, gaes, rewards, v_preds_next]

    return r_sum / T_TIMESTEPS, trans_all_user, success_alls.sum(axis = 0) / T_TIMESTEPS

record_reward = []
loss_episode = []
for i in range(n_episode):
    i_episode = i_episode + 1
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)
    r_avgs = []
    for f in futures:
        r_avg, trans_all_user, success_rate = f.result()
        r_avgs.append(r_avg)
    record_reward.append(sum(r_avgs))

    print('Episode:', i, 'Sum Reward', r_avg)
    loss_batch = []
    sample_indices = np.random.randint(low=0, high=trans_all_user[0].shape[0], size=BATCH_SIZE)
    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in trans_all_user]  # sample training data

    loss_all = []

    for i in range(n_veh):
        for j in range(n_neighbor):
            s = sampled_inp[0][:, i, :]
            a = sampled_inp[1][:, i, :]
            gae = sampled_inp[2][:, i]
            reward = sampled_inp[3][:, i]
            v_pred_next = sampled_inp[4][:, i]

            # s, a, discounted_r = [np.array(e) for e in zip(*trans_with_discounted_r)]
            loss = ppoes.train(s, a, gae, reward, v_pred_next, ppoes.sesses[i * n_neighbor + j])
            # print('Loss_'+'%d:' %i, loss[0])
            # print('Entropy_'+'%d:' %i, entropy)
            loss_all.append(loss[1]) # use entropy to evaluate the stable of the policy
    loss_batch.append(sum(loss_all))
    loss_episode.append(sum(loss_batch)/BATCH_SIZE)
    print('Loss_episode: ', loss_episode[-1])

    if i_episode == int(n_episode / 2):
        label_early = '%d_' % target_average_step + '%d_' % n_veh + '%d_' % i_episode + '%s_' %args.lr_main + '%s_' %args.sigma_add + '%s_' %env_label
        if IS_meta:
            np.savetxt('./Train_data/Reward_AC_'+'%d_' %meta_episode+ label_early, record_reward)
            np.savetxt('./Train_data/loss_AC_'+'%d_' %meta_episode+ label_early, loss_episode)
            ppoes.save_models('PPO_AC_' +'%d_' %meta_episode+ label_early)
        else:
            np.savetxt('./Train_data/Reward_no_meta_AC_' + label_early, record_reward)
            np.savetxt('./Train_data/loss_no_meta_AC_' + label_early, loss_episode)
            ppoes.save_models('PPO_no_meta_AC_' + label_early)

    if i_episode % target_average_step == target_average_step - 1 and i_episode < 0.9 * n_episode:
        print('Model averaged ' + '%d' % current_fed_times)
        current_fed_times = current_fed_times + 1
        ppoes.averaging_model(success_rate)

label_base = '%d_' %target_average_step+ '%d_' %n_veh + '%d_' %n_episode + '%s_' %args.lr_main + '%s_' %args.sigma_add + '%s_' %env_label
if IS_meta:
    np.savetxt('./Train_data/Reward_AC_'+'%d_' %meta_episode+ label_base, record_reward)
    np.savetxt('./Train_data/loss_AC_'+'%d_' %meta_episode+ label_base, loss_episode)
    ppoes.save_models('PPO_AC_' +'%d_' %meta_episode+ label_base)
else:
    np.savetxt('./Train_data/Reward_no_meta_AC_'+ label_base, record_reward)
    np.savetxt('./Train_data/loss_no_meta_AC_'+ label_base, loss_episode)
    ppoes.save_models('PPO_no_meta_AC_' + label_base)
# np.savetxt('loss_analysis', loss_episode)
plt.rcParams['figure.dpi'] = 300
plt.figure(1)
plt.grid()
plt.plot(record_reward)
plt.ylabel('Training Reward')
plt.xlabel('Episodes')
plt.legend()
plt.show()
