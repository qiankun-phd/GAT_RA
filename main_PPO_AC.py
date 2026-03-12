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
IS_FL_adaptive = args.fl_adaptive_interval  # 自适应聚合频率开关
IS_FL_soft = args.fl_soft_aggregation  # 软聚合开关
FL_aggregation_weight = args.fl_aggregation_weight  # 软聚合权重（仅在软聚合启用时使用）
IS_FL_layer_wise = args.fl_layer_wise  # 分层联邦聚合开关
IS_FL_semantic_weighting = args.fl_semantic_weighting  # 语义感知加权开关
FL_semantic_temperature = args.fl_semantic_temperature  # 语义加权温度系数
IS_FL_use_success_rate = getattr(args, 'fl_use_success_rate_weighting', False)  # 是否使用成功率权重（默认False，使用均匀权重）
USE_EPISODE_PROGRESS = getattr(args, 'state_use_episode_progress', True)  # 是否在state中使用episode进度
n_veh = args.n_veh
n_RB = args.n_RB

# 如果要求与backup版本保持一致：关闭所有“改进型FL”开关，使用固定频率+硬替换

# 固定使用SEE优化目标（只优化Semantic-EE）
optimization_target = 'SEE'
# beta参数已废弃，不再使用（保留仅为了向后兼容）
beta = args.beta if hasattr(args, 'beta') else 0.5
circuit_power = args.circuit_power if hasattr(args, 'circuit_power') else 0.06

# Task similarity model: Q = A_peak*(1-e^{-xi*rho})/(1+e^{-zeta*(gamma-gamma0)}) + b (SNR -20~60 dB)
task_sim_A_peak = getattr(args, 'task_sim_A_peak', 0.7128)
task_sim_xi = getattr(args, 'task_sim_xi', 10.0)
task_sim_zeta = getattr(args, 'task_sim_zeta', 0.2313)
task_sim_gamma0 = getattr(args, 'task_sim_gamma0', 0.0)
task_sim_b = getattr(args, 'task_sim_b', 0.3249)
sig2_dB = getattr(args, 'sig2_dB', -160)
area_size = getattr(args, 'area_size', 500.0)
path_loss_offset_dB = getattr(args, 'path_loss_offset_dB', 0.0)
path_loss_model = getattr(args, 'path_loss_model', 'A2G')
env_indoor = Environment_marl_indoor.Environ(
    n_veh, n_RB,
    beta=beta,
    circuit_power=circuit_power,
    optimization_target=optimization_target,
    area_size=area_size,
    task_sim_A_peak=task_sim_A_peak,
    task_sim_xi=task_sim_xi,
    task_sim_zeta=task_sim_zeta,
    task_sim_gamma0=task_sim_gamma0,
    task_sim_b=task_sim_b,
    sig2_dB=sig2_dB,
    path_loss_offset_dB=path_loss_offset_dB,
    path_loss_model=path_loss_model
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

# SINR 归一化范围：训练模拟约 [-103, 60] dB，取可操作区间线性映射到 [0,1]
# 成功门限 3.16 dB -> (3.16 + 30)/100 ≈ 0.33，便于网络区分
SINR_DB_LOW = -30
SINR_DB_HIGH = 70


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
    rho_current = getattr(env, 'rho_current', np.zeros(env.n_Veh))[idx[0]]
    sinr_dB = 10 * np.log10(getattr(env, 'cellular_SINR', np.zeros(env.n_Veh))[idx[0]] + 1e-10)  # dB, avoid log(0)
    sinr_normalized = (sinr_dB - SINR_DB_LOW) / (SINR_DB_HIGH - SINR_DB_LOW)  # [SINR_DB_LOW, SINR_DB_HIGH] -> [0, 1]
    sinr_normalized = np.clip(sinr_normalized, 0.0, 1.0)
    
    # Normalize semantic metrics to [0, 1] (state uses semantic_accuracy only)
    semantic_accuracy_norm = (semantic_accuracy - 0.2) / 0.8 if 0.8 > 0 else semantic_accuracy
    semantic_accuracy_norm = np.clip(semantic_accuracy_norm, 0.0, 1.0)
    semantic_EE_norm = np.log1p(semantic_EE) / np.log1p(10.0)
    semantic_EE_norm = np.clip(semantic_EE_norm, 0.0, 1.0)
    # 与 act_for_training 一致：语义是否满足阈值 (0.5)
    semantic_meets_threshold = 1.0 if semantic_accuracy >= 0.5 else 0.0

    return np.concatenate((
        np.reshape(cellular_fast, -1),      # n_RB维
        np.reshape(cellular_abs, -1),       # n_RB维
        np.reshape(channel_choice, -1),      # n_RB维
        vehicle_vector,                      # n_RB维
        np.asarray([
            success,                         # 1维 - 传输是否成功
            ind_episode / (n_episode) if USE_EPISODE_PROGRESS else 0.0,  # 1维 - episode进度
            semantic_accuracy_norm,          # 1维 - 语义准确度
            semantic_EE_norm,                # 1维 - 语义能量效率
            rho_current,                     # 1维 - 当前压缩比
            sinr_normalized,                 # 1维 - SINR (归一化)
            semantic_meets_threshold         # 1维 - 语义是否满足阈值 (>=0.5)
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

ppoes = PPO(state_dim, action_bound, args.weight_for_L_vf, args.weight_for_entropy, args.epsilon,
            args.lr_main, args.lr_meta_a, args.minibatch_steps, n_veh, n_RB, IS_meta, meta_episode)

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

    state_all = []
    for i in range(n_veh):
        state_all.append(get_state(env, [i, 0], n_veh, i_episode))
    
    for step in range(T_TIMESTEPS):
        env.renew_BS_channels_fastfading()

        action_all = []
        v_pred_all = []
        reward_all = []
        action_all_training = np.zeros([n_veh, 3], dtype='float32')

        for i in range(n_veh):
            state_ = get_state(env, [i, 0], n_veh, i_episode)
            action = ppoes.choose_action(state_, ppoes.sesses[i])
            v_pred = ppoes.get_v(state_, ppoes.sesses[i])
            
            action_all.append(action)
            v_pred_all.append(v_pred)
            # 与 backup 一致：PPO 输出 raw [-1,1]，仅传给 env 时映射到 [0,1]；(a+1)/2
            max_power_dB = env.cellular_power_dB_List[0]
            power_01 = (float(action[1]) + 1.0) * 0.5
            rho_01 = (float(action[2]) + 1.0) * 0.5
            action_all_training[i, 0] = action[0]  # RB
            action_all_training[i, 1] = power_01 * max_power_dB  # Power (dB)
            action_all_training[i, 2] = rho_01  # rho [0,1]
            
        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp, IS_PPO)
        for i in range(n_veh):
            reward_all.append(train_reward)
            state_all[i] = get_state(env, [i, 0], n_veh, i_episode)
        success_all = env.success
        similarity_success_all = env.similarity_success
        r_sum += train_reward

        state_alls = np.append(state_alls, np.asarray(state_all))
        action_alls = np.append(action_alls, np.asarray(action_all))
        v_pred_alls = np.append(v_pred_alls, np.asarray(v_pred_all))
        rewards = np.append(rewards, np.asarray(reward_all))
        success_alls = np.append(success_alls, np.asarray(success_all))
        similarity_success_alls = np.append(similarity_success_alls, np.asarray(similarity_success_all))
        v_preds_next = v_pred_alls

    # Normalize rewards (always, like backup): (r - mean) / (std + 1e-8)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = np.append(v_pred_alls[1:], np.zeros([n_veh]))
    rewards = rewards.reshape([-1, n_veh])
    v_pred_alls = v_pred_alls.reshape([-1, n_veh])
    v_preds_next = v_preds_next.reshape([-1, n_veh])
    success_alls = success_alls.reshape([-1, n_veh])
    similarity_success_alls = similarity_success_alls.reshape([-1, n_veh])  # Reshape similarity success

    gaes = ppoes.get_gaes(rewards=rewards, v_preds=v_pred_alls, v_preds_next=v_preds_next)

    state_alls = np.reshape(state_alls, newshape=(-1, n_veh, state_dim))
    action_alls = np.reshape(action_alls, newshape=(-1, n_veh, action_dim))
    gaes = np.array(gaes).astype(dtype=np.float32)
    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

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
# 模型保存前缀：与 tensorboard 对应，便于区分 MFRL/MRL/FRL/RL
model_prefix = 'PPO_' + training_mode + '_'

log_name_parts = [opt_target_str, algorithm_str]
if training_mode:
    log_name_parts.append(training_mode)
# 添加语义参数到日志名（不再包含reward beta，因为只优化SEE）
log_name_parts.append(f"UAV{n_veh}_RB{n_RB}")
log_name_parts.append(f"area{int(area_size)}")
# 添加学习率到日志名
log_name_parts.append(f"lr{args.lr_main}")
# 策略探索噪声 sigma_add（影响 Reward/loss 保存名与 TensorBoard 区分）
log_name_parts.append(f"sigma_add{args.sigma_add}")
# 如果使用FL噪声初始化，将sigma记录到日志名中，便于区分不同配置
if getattr(args, 'fl_noise_sigma', None) is not None:
    log_name_parts.append(f"Sig{args.fl_noise_sigma}")
# 如果启用联邦学习，添加聚合频率信息和改进方式
if IS_FL:
    # 计算最大可能的聚合次数（在0.9*n_episode之前，每target_average_step步聚合一次）
    max_fed_times = int(0.9 * n_episode / target_average_step)
    
    # 构建FL配置字符串
    fl_config_parts = [f"FL{target_average_step}_max{max_fed_times}"]

    # 添加自适应聚合频率标识
    if IS_FL_adaptive:
        fl_config_parts.append("Adapt")
    
    # 添加软聚合标识
    if IS_FL_soft:
        # 将权重转换为字符串（例如：0.7 -> "S07"）
        weight_str = f"S{int(FL_aggregation_weight * 100):02d}"
        fl_config_parts.append(weight_str)
    
    # 添加分层聚合标识
    if IS_FL_layer_wise:
        fl_config_parts.append("Layer")
    
    # 添加语义感知加权标识
    if IS_FL_semantic_weighting:
        temp_str = f"Sem{int(FL_semantic_temperature * 100):02d}"
        fl_config_parts.append(temp_str)
    
    log_name_parts.append("_".join(fl_config_parts))
if getattr(args, 'experiment_tag', ''):
    log_name_parts.append(args.experiment_tag)
log_name_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
log_name = "_".join(log_name_parts)

_log_base = getattr(args, 'log_dir', 'logs/tensorboard') or 'logs/tensorboard'
log_dir = os.path.join(_log_base, log_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
writer = tf.summary.FileWriter(log_dir)
print(f"TensorBoard日志目录: {log_dir}")
print(f"启动TensorBoard: tensorboard --logdir={_log_base} --port=6008")

lr_decay_after_ratio = getattr(args, 'lr_decay_after_ratio', 0.0)
lr_decay_gamma = getattr(args, 'lr_decay_gamma', 0.5)
# 收敛过程最优模型：label_base / model_label_base 用于保存 _best 与最终模型
label_base = '%d_' % target_average_step + '%d_' % n_veh + '%d_' % n_episode + '%s_' % args.lr_main + '%s_' % args.sigma_add + '%s_' % env_label
model_save_dir = getattr(args, 'model_save_dir', '') or ''
_model_label_base = model_prefix + ('%d_' % meta_episode if IS_meta else '') + label_base
model_label_base = (model_save_dir + '/' if model_save_dir else '') + _model_label_base
best_rolling_reward = -np.inf
save_best_rolling = getattr(args, 'save_best_rolling', 50)
save_best_enabled = getattr(args, 'save_best', True) and not getattr(args, 'no_save_best', False)

for episode_idx in range(n_episode):
    i_episode = i_episode + 1
    # 学习率线性衰减（防策略退步）：从 lr_decay_after_ratio 起线性降到 lr_main*gamma
    if lr_decay_after_ratio > 0 and i_episode >= n_episode * lr_decay_after_ratio:
        start_ep = int(n_episode * lr_decay_after_ratio)
        progress = (i_episode - start_ep) / max(1, n_episode - start_ep)  # 0 -> 1
        current_lr = args.lr_main * (lr_decay_gamma + (1.0 - lr_decay_gamma) * (1.0 - progress))
    else:
        current_lr = args.lr_main

    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)
    r_avgs = []
    for f in futures:
        r_avg, trans_all_user, success_rate, similarity_rate = f.result()
        r_avgs.append(r_avg)
    record_reward.append(sum(r_avgs))

    print('Episode:', episode_idx, 'Sum Reward', r_avg, ' Channel Success Rate', success_rate, 'Similarity Success Rate', similarity_rate)
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

        loss = ppoes.train(s, a, gae, reward, v_pred_next, ppoes.sesses[agent_idx], lr=current_lr)
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
    summary.value.add(tag='Train/reward', simple_value=float(r_avg)*n_veh)  # Total SEE (reward 为 per-UAV 平均)
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
    
    if lr_decay_after_ratio > 0:
        summary.value.add(tag='Train/lr', simple_value=float(current_lr))
    writer.add_summary(summary, i_episode)
    writer.flush()

    # 收敛过程最优模型：按滚动平均 reward 保存 _best
    if save_best_enabled and save_best_rolling > 0 and len(record_reward) >= save_best_rolling:
        rolling_avg = np.mean(record_reward[-save_best_rolling:])
        if rolling_avg > best_rolling_reward:
            best_rolling_reward = rolling_avg
            model_label_best = model_label_base.rstrip('_') + '_best'
            ppoes.save_models(model_label_best)
            print('  [Save best] rolling reward %.4f -> %s' % (rolling_avg, model_label_best))

    if getattr(args, 'mid_episode_save', True) and i_episode == int(n_episode / 2):
        label_early = '%d_' % target_average_step + '%d_' % n_veh + '%d_' % i_episode + '%s_' % args.lr_main + '%s_' % args.sigma_add + '%s_' % env_label
        _model_label_early = model_prefix + ('%d_' % meta_episode if IS_meta else '') + label_early
        model_label_early = (model_save_dir + '/' if model_save_dir else '') + _model_label_early
        _rl_safe = (model_save_dir.replace('/', '_') + '_' if model_save_dir else '')  # 避免 / 导致 Train_data 子目录
        _rl_base = _rl_safe + _model_label_early.rstrip('_')
        np.savetxt('./Train_data/Reward_' + _rl_base + '_seed%d' % args.seed, record_reward)
        np.savetxt('./Train_data/loss_' + _rl_base + '_seed%d' % args.seed, loss_episode)
        ppoes.save_models(model_label_early)

    # 联邦学习：模型平均
    # 自适应聚合频率：根据训练阶段调整聚合间隔
    if IS_FL:
        # 自适应聚合频率：根据训练阶段调整聚合间隔
        if IS_FL_adaptive:
            # 自适应聚合频率：阶段化训练
            if i_episode < n_episode * 0.3:
                # 早期（前30%）：较少聚合，让元学习充分发挥
                aggregation_interval = target_average_step * 2  # 200步
                stage = "early"
            elif i_episode < n_episode * 0.7:
                # 中期（30%-70%）：正常聚合
                aggregation_interval = target_average_step  # 100步
                stage = "mid"
            else:
                # 后期（70%-90%）：更频繁聚合，利用协作优势
                aggregation_interval = max(target_average_step // 2, 10)  # 50步（至少10步）
                stage = "late"
        else:
            # 原有逻辑：固定聚合频率
            aggregation_interval = target_average_step
            stage = "fixed"
        
        # 检查是否到达聚合时机
        # 说明：
        # - 之前用 i_episode < 0.9 * n_episode 会导致 n_episode 较小时（如 50）完全不触发聚合
        # - 这里保证：至少能在一个 interval 内触发 1 次聚合（例如 i_episode==aggregation_interval-1）
        agg_cutoff = max(int(0.9 * n_episode), aggregation_interval)
        should_aggregate = (i_episode % aggregation_interval == aggregation_interval - 1) and (i_episode < agg_cutoff)
        
        if should_aggregate:
            # 计算聚合权重
            external_weights = None
            if IS_FL_semantic_weighting:
                # 1. 提取本轮次各 UAV 的平均语义能效 (Semantic EE)
                # trans_all_user[0] 是 state_alls, shape=[Steps, n_veh, state_dim]
                # semantic_EE_norm 是状态向量中倒数第4个元素 (index -4)
                states_history = trans_all_user[0]  # Shape: [Steps, n_veh, state_dim]
                
                # 计算每个 Agent 在本局的平均 Semantic EE
                avg_see_per_agent = np.mean(states_history[:, :, -4], axis=0)  # Shape: [n_veh]
                
                # 2. 计算语义权重 (Semantic Weights) - 使用 Softmax 放大优势
                # 温度系数 T: 越小，好用户权重越大；越大，权重越平均
                T = FL_semantic_temperature
                exp_see = np.exp((avg_see_per_agent - np.max(avg_see_per_agent)) / T)
                external_weights = exp_see / np.sum(exp_see)
                
                print(f"🧠 语义感知加权: Avg SEE={np.round(avg_see_per_agent, 3)}, Weights={np.round(external_weights, 3)}")
            
            if IS_FL_adaptive:
                print(f'Model averaged (stage: {stage}, interval: {aggregation_interval}) ' + '%d' % current_fed_times)
            else:
                print('Model averaged ' + '%d' % current_fed_times)
            current_fed_times = current_fed_times + 1
            
            # 传递语义感知权重和软聚合参数
            # 注意：默认行为即为“backup式”累加器噪声初始化（在 averaging_model 内部实现）
            if IS_FL_soft:
                print(f"💡 调用averaging_model: IS_FL_soft=True, aggregation_weight={FL_aggregation_weight}, layer_wise={IS_FL_layer_wise}, external_weights={'provided' if external_weights is not None else 'None'}, use_success_rate_weighting={IS_FL_use_success_rate}")
                ppoes.averaging_model(success_rate, aggregation_weight=FL_aggregation_weight, layer_wise=IS_FL_layer_wise, external_weights=external_weights, use_success_rate_weighting=IS_FL_use_success_rate)
            else:
                # 硬替换（aggregation_weight=1.0），累加器使用随机初始化（与backup一致）
                print(f"⚙️  调用averaging_model: IS_FL_soft=False, aggregation_weight=1.0, layer_wise={IS_FL_layer_wise}, external_weights={'provided' if external_weights is not None else 'None'}, use_success_rate_weighting={IS_FL_use_success_rate}")
                ppoes.averaging_model(success_rate, aggregation_weight=1.0, layer_wise=IS_FL_layer_wise, external_weights=external_weights, use_success_rate_weighting=IS_FL_use_success_rate)

# 最终模型：保存到 base 名称（最后一轮）；收敛过程最优已保存为 _best；Reward/loss 带种子标识避免多种子覆盖
model_label = model_label_base
_reward_loss_suffix = '_seed%d' % args.seed
_rl_safe_final = (model_save_dir.replace('/', '_') + '_' if model_save_dir else '')  # 避免 / 导致 Train_data 子目录
_rl_base_final = _rl_safe_final + _model_label_base.rstrip('_')
np.savetxt('./Train_data/Reward_' + _rl_base_final + _reward_loss_suffix, record_reward)
np.savetxt('./Train_data/loss_' + _rl_base_final + _reward_loss_suffix, loss_episode)
ppoes.save_models(model_label)
print('  [Save last] final model -> %s, curves -> Reward_/loss_*%s' % (model_label.rstrip('_'), _reward_loss_suffix))

writer.close()
print(f"\n训练完成！TensorBoard日志已保存到: {log_dir}")
print(f"查看TensorBoard: tensorboard --logdir={_log_base} --port=6008")

