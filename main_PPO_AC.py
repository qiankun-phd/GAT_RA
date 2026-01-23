import random
import concurrent.futures
import numpy as np
import os
from PPO_brain_AC import PPO
import matplotlib.pyplot as plt
import Environment_marl_indoor
# import Environment_marl_urban_micro
# import Environment_marl_urban_macro
# import Environment_marl_rural_macro

from arguments import get_args
args = get_args()
import random
from tensorflow.compat.v1 import set_random_seed
import tensorflow.compat.v1 as tf
from datetime import datetime

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
set_random_seed(args.set_random_seed)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

meta_episode = args.meta_episode
target_average_step = args.target_average_step

IS_PPO = args.IS_PPO
IS_meta = args.Do_meta
IS_FL = args.Do_FL
n_veh = args.n_veh
n_RB = args.n_RB

# 固定使用SEE优化目标（只优化Semantic-EE）
optimization_target = 'SEE'
# beta参数已废弃，不再使用（保留仅为了向后兼容）
beta = args.beta if hasattr(args, 'beta') else 0.5
circuit_power = args.circuit_power if hasattr(args, 'circuit_power') else 0.06

# Get semantic communication parameters
semantic_A_max = args.semantic_A_max if hasattr(args, 'semantic_A_max') else 1.0
semantic_beta = args.semantic_beta if hasattr(args, 'semantic_beta') else 2.0
# Sigmoid model parameters
semantic_A1 = args.semantic_A1 if hasattr(args, 'semantic_A1') else 1.0
semantic_A2 = args.semantic_A2 if hasattr(args, 'semantic_A2') else 0.2
semantic_C1 = args.semantic_C1 if hasattr(args, 'semantic_C1') else 5.0
semantic_C2 = args.semantic_C2 if hasattr(args, 'semantic_C2') else 2.0

env_indoor = Environment_marl_indoor.Environ(
    n_veh, n_RB, 
    beta=beta, 
    circuit_power=circuit_power, 
    optimization_target=optimization_target,
    semantic_A_max=semantic_A_max,
    semantic_beta=semantic_beta,
    semantic_A1=semantic_A1,
    semantic_A2=semantic_A2,
    semantic_C1=semantic_C1,
    semantic_C2=semantic_C2
)

env_choice = args.env_choice
if env_choice == 0:
    env = env_indoor
    env_label = "indoor"
env.new_random_game()

GAMMA = args.gamma
BATCH_SIZE = args.meta_batch_size
i_episode = 0
n_episode = args.n_episode
ACTOR_NUM = 1
T_TIMESTEPS = int(env.time_slow / (env.time_fast))
current_fed_times = 0


def get_state(env, idx=(0, 0), n_veh=0, ind_episode=0.):
    """ Get state from the environment (for MLP mode) """
    cellular_fast = (env.cellular_channels_with_fastfading[idx[0], :] - env.cellular_channels_abs[idx[0]] + 10) / 35
    cellular_abs = (env.cellular_channels_abs[idx[0]] - 80) / 60.0
    success = env.success[idx[0]]
    channel_choice = env.channel_choice / max(n_veh, 1)  # Avoid division by zero
    vehicle_vector = np.zeros(n_RB)
    for i in range(n_veh):
        vehicle_vector[i] = 1 / n_veh
    
    # Semantic communication metrics (normalized)
    semantic_accuracy = getattr(env, 'semantic_accuracy', np.zeros(env.n_Veh))[idx[0]]
    semantic_EE = getattr(env, 'semantic_EE', np.zeros(env.n_Veh))[idx[0]]
    semantic_similarity = getattr(env, 'semantic_similarity', np.zeros(env.n_Veh))[idx[0]]
    rho_current = getattr(env, 'rho_current', np.zeros(env.n_Veh))[idx[0]]
    sinr_dB = 10 * np.log10(getattr(env, 'cellular_SINR', np.zeros(env.n_Veh))[idx[0]] + 1e-10)  # Convert to dB, avoid log(0)
    sinr_normalized = (sinr_dB + 20) / 40.0  # Normalize: assume range [-20, 20] dB -> [0, 1]
    sinr_normalized = np.clip(sinr_normalized, 0.0, 1.0)
    
    # Normalize semantic metrics to [0, 1] range
    # semantic_accuracy is already in [A2, A1] = [0.2, 1.0], normalize to [0, 1]
    semantic_accuracy_norm = (semantic_accuracy - 0.2) / 0.8 if 0.8 > 0 else semantic_accuracy
    semantic_accuracy_norm = np.clip(semantic_accuracy_norm, 0.0, 1.0)
    
    # semantic_EE can vary widely, use log normalization
    semantic_EE_norm = np.log1p(semantic_EE) / np.log1p(10.0)  # Normalize assuming max ~10
    semantic_EE_norm = np.clip(semantic_EE_norm, 0.0, 1.0)
    
    # semantic_similarity is already in [A2, A1] = [0.2, 1.0], normalize to [0, 1]
    semantic_similarity_norm = (semantic_similarity - 0.2) / 0.8 if 0.8 > 0 else semantic_similarity
    semantic_similarity_norm = np.clip(semantic_similarity_norm, 0.0, 1.0)
    
    # rho_current is already in [0, 1], no normalization needed
    
    return np.concatenate((
        np.reshape(cellular_fast, -1),      # n_RB维
        np.reshape(cellular_abs, -1),       # n_RB维
        np.reshape(channel_choice, -1),      # n_RB维
        vehicle_vector,                      # n_RB维
        np.asarray([
            success,                         # 1维
            ind_episode / (n_episode),       # 1维
            semantic_accuracy_norm,          # 1维 - 新增：语义准确度
            semantic_EE_norm,                # 1维 - 新增：语义能量效率
            semantic_similarity_norm,        # 1维 - 新增：语义相似度
            rho_current,                     # 1维 - 新增：当前压缩比
            sinr_normalized                  # 1维 - 新增：SINR (归一化)
        ])
    ))


def save_models(sess, model_path, saver):
    """ Save models to the current directory with the name filename """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)

state_dim = len(get_state(env=env))
action_dim = 3  # RB_choice + power + rho (compression ratio)
action_bound = []
action_bound.append(n_RB)
action_bound.append(args.RB_action_bound)
action_bound.append(1.0)  # rho ∈ [0, 1]

ppoes = []
# Get GAT parameters
use_gat = False
if hasattr(args, 'use_gat') and args.use_gat:
    print("[WARN] 本版本已移除 PPO 内的 GAT 编码器实现，将强制 use_gat=False（继续使用 MLP）。")
num_gat_heads = args.num_gat_heads if hasattr(args, 'num_gat_heads') else 4
node_feature_dim = None  # (GAT已移除) 保留占位以兼容旧参数

ppoes = PPO(state_dim, action_bound, args.weight_for_L_vf, args.weight_for_entropy, args.epsilon, 
            args.lr_main, args.lr_meta_a, args.minibatch_steps, n_veh, n_RB, IS_meta, meta_episode,
            use_gat=use_gat, num_gat_heads=num_gat_heads, node_feature_dim=node_feature_dim)

executor = concurrent.futures.ThreadPoolExecutor(ACTOR_NUM)


def simulate():
    env.renew_positions()
    env.renew_BS_channel()
    env.renew_BS_channels_fastfading()
    r_sum = 0
    trans_all_user = []
    success_alls = []
    similarity_success_alls = []  # Track similarity threshold achievement
    state_alls = []
    action_alls = []
    v_pred_alls = []
    rewards = []

    # Prepare state/graph data based on mode
    if use_gat:
        # GAT mode: get graph data
        node_features, adj_matrix = get_graph_data(env, n_veh, i_episode)
        node_features_all = []  # Store graph data for each step
        adj_matrix_all = []  # Store adjacency matrices for each step
    else:
        # MLP mode: get individual states
        state_all = []
        for i in range(n_veh):
            state_all.append(get_state(env, [i, 0], n_veh, i_episode))
    
    for step in range(T_TIMESTEPS):
        env.renew_BS_channels_fastfading()
        
        # Update graph data if in GAT mode
        if use_gat:
            node_features, adj_matrix = get_graph_data(env, n_veh, i_episode)
            node_features_all.append(node_features)
            adj_matrix_all.append(adj_matrix)
        
        action_all = []
        v_pred_all = []
        reward_all = []
        action_all_training = np.zeros([n_veh, 3], dtype='float32')  # 改为3列

        for i in range(n_veh):
            if use_gat:
                # GAT mode: use graph data
                action = ppoes.choose_action(None, ppoes.sesses[i], 
                                            node_features=node_features, 
                                            adj_matrix=adj_matrix, 
                                            agent_idx=i)
                v_pred = ppoes.get_v(None, ppoes.sesses[i],
                                    node_features=node_features,
                                    adj_matrix=adj_matrix,
                                    agent_idx=i)
                # For state storage (backward compatibility)
                state_ = get_state(env, [i, 0], n_veh, i_episode)
            else:
                # MLP mode: use individual state
                state_ = get_state(env, [i, 0], n_veh, i_episode)
                action = ppoes.choose_action(state_, ppoes.sesses[i])
                v_pred = ppoes.get_v(state_, ppoes.sesses[i])
            
            action_all.append(action)
            v_pred_all.append(v_pred)
            amp = env.cellular_power_dB_List[0] / (2 * action_bound[1])
            power_action = (action[1] + action_bound[1]) * amp
            action_all_training[i, 0] = action[0]  # RB
            action_all_training[i, 1] = power_action  # Power
            action_all_training[i, 2] = action[2]  # rho (compression ratio)
            
        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp, IS_PPO)
        for i in range(n_veh):
            reward_all.append(train_reward)
            if not use_gat:
                state_ = get_state(env, [i, 0], n_veh, i_episode)
                state_all[i] = state_
        success_all = env.success
        similarity_success_all = env.similarity_success  # Get similarity threshold achievement
        r_sum += train_reward

        if not use_gat:
            state_alls = np.append(state_alls, np.asarray(state_all))
        action_alls = np.append(action_alls, np.asarray(action_all))
        v_pred_alls = np.append(v_pred_alls, np.asarray(v_pred_all))
        rewards = np.append(rewards, np.asarray(reward_all))
        success_alls = np.append(success_alls, np.asarray(success_all))
        similarity_success_alls = np.append(similarity_success_alls, np.asarray(similarity_success_all))
        v_preds_next = v_pred_alls

    # Normalize rewards only if std > threshold (avoid dividing by zero)
    rewards_std = rewards.std()
    if rewards_std > 1e-8:
        rewards = (rewards - rewards.mean()) / rewards_std
    # If all rewards are the same, keep original values (don't normalize to zero)
    
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = np.append(v_pred_alls[1:], np.zeros([n_veh]))
    rewards = rewards.reshape([-1, n_veh])
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = v_preds_next.reshape([-1, n_veh])
    success_alls = success_alls.reshape([-1, n_veh])
    similarity_success_alls = similarity_success_alls.reshape([-1, n_veh])  # Reshape similarity success

    gaes = ppoes.get_gaes(rewards=rewards, v_preds=v_pred_alls, v_preds_next=v_preds_next)

    if use_gat:
        # GAT mode: prepare graph data for training
        node_features_batch = np.array(node_features_all)  # [T_TIMESTEPS, n_veh, node_feature_dim]
        adj_matrix_batch = np.array(adj_matrix_all)  # [T_TIMESTEPS, n_veh, n_veh]
        # For backward compatibility, also create state_alls from node_features
        # Flatten to [T_TIMESTEPS * n_veh, node_feature_dim] then reshape
        state_alls = node_features_batch.reshape([-1, node_features_batch.shape[-1]])
        state_alls = np.reshape(state_alls, newshape=(-1, n_veh, node_features_batch.shape[-1]))
    else:
        # MLP mode: reshape states
        state_alls = np.reshape(state_alls, newshape=(-1, n_veh, state_dim))
    
    action_alls = np.reshape(action_alls, newshape=(-1, n_veh, action_dim))
    gaes = np.array(gaes).astype(dtype=np.float32)
    gaes = (gaes - gaes.mean()) / gaes.std()

    if use_gat:
        trans_all_user = [state_alls, action_alls, gaes, rewards, v_preds_next, node_features_batch, adj_matrix_batch]
    else:
        trans_all_user = [state_alls, action_alls, gaes, rewards, v_preds_next]

    success_rate = success_alls.sum(axis=0) / T_TIMESTEPS
    similarity_rate = similarity_success_alls.sum(axis=0) / T_TIMESTEPS  # Calculate similarity threshold achievement rate
    return r_sum / T_TIMESTEPS, trans_all_user, success_rate, similarity_rate

record_reward = []
loss_episode = []

# 初始化TensorBoard
opt_target_str = "SEE"  # 固定为SEE（Semantic-EE）
algorithm_str = "MAPPO"

if IS_FL and IS_meta:
    training_mode = "MFRL"
elif IS_FL:
    training_mode = "FRL"
elif IS_meta:
    training_mode = "MRL"
else:
    training_mode = "RL"

log_name_parts = [opt_target_str, algorithm_str]
if training_mode:
    log_name_parts.append(training_mode)
# 添加语义参数到日志名（不再包含reward beta，因为只优化SEE）
# 格式: Amax1.0_semB2.0 (语义A_max, 语义beta)
log_name_parts.append(f"Amax{semantic_A_max}_semB{semantic_beta}")
log_name_parts.append(f"UAV{n_veh}_RB{n_RB}")
# 添加学习率到日志名
log_name_parts.append(f"lr{args.lr_main}")
# 如果启用联邦学习，添加聚合频率信息
if IS_FL:
    # 计算最大可能的聚合次数（在0.9*n_episode之前，每target_average_step步聚合一次）
    max_fed_times = int(0.9 * n_episode / target_average_step)
    log_name_parts.append(f"FL{target_average_step}_max{max_fed_times}")
log_name = "_".join(log_name_parts)

log_dir = f'./logs/tensorboard/{log_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
writer = tf.summary.FileWriter(log_dir)
print(f"TensorBoard日志目录: {log_dir}")
print(f"启动TensorBoard: tensorboard --logdir=./logs/tensorboard --port=6008")

for episode_idx in range(n_episode):
    i_episode = i_episode + 1
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)
    r_avgs = []
    for f in futures:
        r_avg, trans_all_user, success_rate, similarity_rate = f.result()
        r_avgs.append(r_avg)
    record_reward.append(sum(r_avgs))

    print('Episode:', episode_idx, 'Sum Reward', r_avg, 'Success Rate', success_rate, 'Similarity Rate', similarity_rate)
    loss_batch = []
    sample_indices = np.random.randint(low=0, high=trans_all_user[0].shape[0], size=BATCH_SIZE)
    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in trans_all_user]

    loss_all = []
    policy_losses = []
    vf_losses = []
    entropies = []

    for agent_idx in range(n_veh):
        s = sampled_inp[0][:, agent_idx, :]
        a = sampled_inp[1][:, agent_idx, :]
        gae = sampled_inp[2][:, agent_idx]
        reward = sampled_inp[3][:, agent_idx]
        v_pred_next = sampled_inp[4][:, agent_idx]

        if use_gat:
            # GAT mode: pass graph data
            node_features_batch = sampled_inp[5]  # [batch_size, n_veh, node_feature_dim]
            adj_matrix_batch = sampled_inp[6]  # [batch_size, n_veh, n_veh]
            # Debug: check data shapes and values
            if agent_idx == 0 and episode_idx < 2:
                print(f"DEBUG Training: reward shape={reward.shape}, reward sample={reward[:3]}, "
                      f"v_pred_next shape={v_pred_next.shape}, v_pred_next sample={v_pred_next[:3]}, "
                      f"gae shape={gae.shape}, gae sample={gae[:3]}")
            loss = ppoes.train(s, a, gae, reward, v_pred_next, ppoes.sesses[agent_idx],
                             node_features=node_features_batch, adj_matrix=adj_matrix_batch, agent_idx=agent_idx)
        else:
            # MLP mode: use state only
            loss = ppoes.train(s, a, gae, reward, v_pred_next, ppoes.sesses[agent_idx])
        # loss[0] = [L_clip, L_RB, L_rho, L_vf, S]
        if len(loss[0]) >= 5:
            policy_losses.append(loss[0][0] + loss[0][1] + loss[0][2])  # L_clip + L_RB + L_rho
            vf_losses.append(loss[0][3])  # L_vf
            entropies.append(loss[0][4])  # S
        loss_all.append(loss[1])
    loss_batch.append(sum(loss_all))
    loss_episode.append(sum(loss_batch)/BATCH_SIZE)
    print('Loss_episode: ', loss_episode[-1])
    
    # 记录到TensorBoard
    summary = tf.Summary()
    summary.value.add(tag='Train/reward', simple_value=float(r_avg)*6)
    summary.value.add(tag='Train/Loss_episode', simple_value=float(loss_episode[-1]))
    
    # Success Rate
    if isinstance(success_rate, np.ndarray):
        summary.value.add(tag='Metrics/success_rate_mean', simple_value=float(np.mean(success_rate)))
        for idx, rate in enumerate(success_rate):
            summary.value.add(tag=f'Metrics/success_rate_ue_{idx}', simple_value=float(rate))
    else:
        summary.value.add(tag='Metrics/success_rate', simple_value=float(success_rate))
    
    # Similarity Threshold Achievement Rate
    if isinstance(similarity_rate, np.ndarray):
        summary.value.add(tag='Metrics/similarity_rate_mean', simple_value=float(np.mean(similarity_rate)))
        for idx, rate in enumerate(similarity_rate):
            summary.value.add(tag=f'Metrics/similarity_rate_ue_{idx}', simple_value=float(rate))
    else:
        summary.value.add(tag='Metrics/similarity_rate', simple_value=float(similarity_rate))
    
    writer.add_summary(summary, i_episode)
    writer.flush()

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

    # 联邦学习：模型平均
    if IS_FL and i_episode % target_average_step == target_average_step - 1 and i_episode < 0.9 * n_episode:
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

writer.close()
print(f"\n训练完成！TensorBoard日志已保存到: {log_dir}")
print(f"查看TensorBoard: tensorboard --logdir=./logs/tensorboard --port=6008")

