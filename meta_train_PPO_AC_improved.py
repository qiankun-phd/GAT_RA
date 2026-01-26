import random
import concurrent.futures
import numpy as np
import os
import time
from collections import defaultdict
from meta_brain_PPO_improved import PPOImproved
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

# 训练配置
meta_episode = args.meta_episode
IS_PPO = args.IS_PPO if hasattr(args, 'IS_PPO') else True
n_veh = args.n_veh_list
n_RB = args.n_RB
sigma_add = args.sigma_add
BATCH_SIZE = args.meta_batch_size
i_episode = 0
ACTOR_NUM = 1
current_fed_times = 0

# Meta学习改进配置
USE_MULTI_STEP_ADAPTATION = True  # 是否使用多步适应
INNER_LOOP_STEPS = 3  # 内循环步数
USE_TASK_WEIGHTS = True  # 是否使用任务权重
USE_TASK_SPECIFIC_NORMALIZATION = True  # 是否使用任务特定归一化
SAVE_CHECKPOINT_EVERY = 50  # 每N个episode保存检查点
EARLY_STOPPING_PATIENCE = 100  # 早停耐心值

def get_state(env, idx=(0, 0), n_veh=1, ind_episode=0.):
    """ Get state from the environment (updated to include semantic communication metrics) """
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
    semantic_accuracy_norm = (semantic_accuracy - 0.2) / 0.8 if 0.8 > 0 else semantic_accuracy
    semantic_accuracy_norm = np.clip(semantic_accuracy_norm, 0.0, 1.0)
    
    semantic_EE_norm = np.log1p(semantic_EE) / np.log1p(10.0)  # Normalize assuming max ~10
    semantic_EE_norm = np.clip(semantic_EE_norm, 0.0, 1.0)
    
    semantic_similarity_norm = (semantic_similarity - 0.2) / 0.8 if 0.8 > 0 else semantic_similarity
    semantic_similarity_norm = np.clip(semantic_similarity_norm, 0.0, 1.0)
    
    return np.concatenate((
        np.reshape(cellular_fast, -1),      # n_RB维
        np.reshape(cellular_abs, -1),       # n_RB维
        np.reshape(channel_choice, -1),      # n_RB维
        vehicle_vector,                      # n_RB维
        np.asarray([
            success,                         # 1维
            ind_episode / (meta_episode),    # 1维
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
    return model_path

def compute_task_weights(n_veh_list, method='complexity'):
    """
    计算任务权重
    method: 'complexity' - 基于任务复杂度, 'equal' - 平等权重, 'performance' - 基于性能
    """
    if method == 'equal':
        return np.ones(len(n_veh_list)) / len(n_veh_list)
    
    elif method == 'complexity':
        # 基于UAV数量的复杂度权重（UAV数量越多，权重越高）
        complexity = np.array(n_veh_list, dtype=float)
        weights = complexity / np.sum(complexity)
        return weights
    
    elif method == 'inverse_complexity':
        # 反向复杂度权重（给简单任务更高权重，帮助快速收敛）
        complexity = np.array(n_veh_list, dtype=float)
        inv_complexity = 1.0 / complexity
        weights = inv_complexity / np.sum(inv_complexity)
        return weights
    
    else:
        raise ValueError(f"Unknown task weighting method: {method}")

def normalize_rewards_per_task(rewards_all_tasks, method='task_specific'):
    """
    任务特定的奖励归一化
    """
    if method == 'task_specific':
        # 每个任务独立归一化
        normalized_rewards = []
        for task_rewards in rewards_all_tasks:
            task_rewards = np.array(task_rewards)
            mean = np.mean(task_rewards)
            std = np.std(task_rewards) + 1e-8
            normalized = (task_rewards - mean) / std
            normalized_rewards.append(normalized)
        return normalized_rewards
    
    elif method == 'global':
        # 全局归一化（原始方法）
        all_rewards = np.concatenate([np.array(rewards).flatten() for rewards in rewards_all_tasks])
        mean = np.mean(all_rewards)
        std = np.std(all_rewards) + 1e-8
        
        normalized_rewards = []
        for task_rewards in rewards_all_tasks:
            normalized = (np.array(task_rewards) - mean) / std
            normalized_rewards.append(normalized)
        return normalized_rewards
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class MetaTrainingStatistics:
    """训练统计信息收集器"""
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.reset()
    
    def reset(self):
        self.task_rewards = [[] for _ in range(self.n_tasks)]
        self.task_losses = [[] for _ in range(self.n_tasks)]
        self.task_gradient_norms = [[] for _ in range(self.n_tasks)]
        self.episode_rewards = []
        self.episode_losses = []
    
    def update_task_stats(self, task_id, reward, loss, gradient_norm):
        self.task_rewards[task_id].append(reward)
        self.task_losses[task_id].append(loss)
        self.task_gradient_norms[task_id].append(gradient_norm)
    
    def update_episode_stats(self, total_reward, total_loss):
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(total_loss)
    
    def get_task_statistics(self, task_id):
        return {
            'mean_reward': np.mean(self.task_rewards[task_id]) if self.task_rewards[task_id] else 0,
            'mean_loss': np.mean(self.task_losses[task_id]) if self.task_losses[task_id] else 0,
            'mean_gradient_norm': np.mean(self.task_gradient_norms[task_id]) if self.task_gradient_norms[task_id] else 0
        }

# 初始化环境和代理
ppoes = []
envs = []
sess = tf.Session(config=my_config)

# 固定使用SEE优化目标
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

T_TIMESTEPS = int(envs[0].time_slow / (envs[0].time_fast))
state_dim = len(get_state(env=envs[0]))
action_dim = 3  # RB_choice + power + rho (compression ratio)
action_bound = []
action_bound.append(n_RB)  # RB bound
action_bound.append(args.RB_action_bound)  # Power bound

# 创建改进的PPO代理（带任务嵌入）
ppo = PPOImproved(state_dim, action_bound, args.weight_for_L_vf, args.weight_for_entropy, 
                  args.epsilon, args.lr_meta_a, args.lr_meta_c, args.minibatch_steps, n_RB, sess, task_id=0)

executor = concurrent.futures.ThreadPoolExecutor(ACTOR_NUM)

# TensorBoard设置
log_dir = f"./logs/meta_training_{time.strftime('%Y%m%d_%H%M%S')}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

# 初始化训练统计
statistics = MetaTrainingStatistics(len(n_veh))
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
task_weights = compute_task_weights(n_veh, method='complexity')

print(f"Meta训练改进版本开始")
print(f"任务配置: {n_veh} UAVs")
print(f"任务权重: {task_weights}")
print(f"多步适应: {'启用' if USE_MULTI_STEP_ADAPTATION else '禁用'} (内循环步数: {INNER_LOOP_STEPS})")
print(f"任务特定归一化: {'启用' if USE_TASK_SPECIFIC_NORMALIZATION else '禁用'}")
print(f"TensorBoard日志目录: {log_dir}")

def simulate():
    """
    改进的模拟函数 - 支持任务特定采样和归一化
    """
    trans_all_user_task = []
    r_sum_task = []

    for k in range(len(n_veh)):
        envs[k].renew_positions()
        envs[k].renew_BS_channel()
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
            envs[k].renew_BS_channels_fastfading()
            
            action_all = []
            v_pred_all = []
            reward_all = []
            action_all_training = np.zeros([n_veh[k], 3], dtype='float32')

            for i in range(n_veh[k]):
                # 使用任务嵌入
                task_embedding = float(n_veh[k]) / 10.0  # 归一化的任务嵌入
                action = ppo.choose_action(state_all[i], sess, task_embedding)
                v_pred = ppo.get_v(state_all[i], sess, task_embedding).tolist()
                action_all.append(action)

                channel_action = action[0]
                amp = envs[k].cellular_power_dB_List[0] / (2 * action_bound[1])
                power_action = (action[1] + action_bound[1]) * amp
                rho_action = action[2]  # 已经是[0,1]范围内的值（Beta分布输出）
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

        v_pred_alls = v_pred_alls.reshape([-1, n_veh[k]])
        v_preds_next = np.append(v_pred_alls[1:], np.zeros([n_veh[k]]))
        rewards = rewards.reshape([-1, n_veh[k]])
        v_pred_alls = v_pred_alls.reshape([-1, n_veh[k]])
        v_preds_next = v_preds_next.reshape([-1, n_veh[k]])

        gaes = ppo.get_gaes(rewards=rewards, v_preds=v_pred_alls, v_preds_next=v_preds_next)
        gaes = np.array(gaes).astype(dtype=np.float32)

        state_alls = np.reshape(state_alls, newshape=(-1, n_veh[k], state_dim))
        action_alls = np.reshape(action_alls, newshape=(-1, n_veh[k], action_dim))

        trans_all_user = [state_alls, action_alls, gaes, rewards, v_preds_next]

        r_sum_task.append(r_sum / T_TIMESTEPS)
        trans_all_user_task.append(trans_all_user)

    return r_sum_task, trans_all_user_task

# 训练主循环
record_reward = []
loss_episode = []
best_loss = float('inf')
no_improvement_count = 0

for episode in range(meta_episode):
    episode_start_time = time.time()
    i_episode = episode + 1
    
    # 收集数据
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)

    r_avgs = []
    all_trans_data = []
    for f in futures:
        r_avg_task, trans_all_user_task = f.result()
        for k in range(len(n_veh)):
            r_avgs.append(r_avg_task[k])
        all_trans_data.extend(trans_all_user_task)

    record_reward.append(np.sum(r_avgs))
    
    # 任务特定归一化奖励（如果启用）
    if USE_TASK_SPECIFIC_NORMALIZATION:
        rewards_all_tasks = []
        for k in range(len(n_veh)):
            task_rewards = all_trans_data[k][3]  # rewards
            rewards_all_tasks.append(task_rewards)
        
        normalized_rewards = normalize_rewards_per_task(rewards_all_tasks, method='task_specific')
        
        # 更新trans_data中的奖励
        for k in range(len(n_veh)):
            all_trans_data[k][3] = normalized_rewards[k]  # 更新归一化后的奖励

    # 训练网络
    loss_batch = []
    total_gradient_norm = 0
    
    for k in range(len(n_veh)):
        # 任务特定采样
        task_data = all_trans_data[k]
        data_size = task_data[0].shape[0]
        task_batch_size = min(BATCH_SIZE, data_size)
        
        # 为每个任务生成独立的采样索引
        task_sample_indices = np.random.randint(low=0, high=data_size, size=task_batch_size)
        sampled_data = [np.take(a=a, indices=task_sample_indices, axis=0) for a in task_data]
        
        task_loss_all = []
        task_embedding = float(n_veh[k]) / 10.0  # 任务嵌入
        
        for i in range(n_veh[k]):
            s = sampled_data[0][:, i, :]  # states
            a = sampled_data[1][:, i, :]  # actions
            gae = sampled_data[2][:, i]   # GAE
            reward = sampled_data[3][:, i]  # rewards
            v_pred_next = sampled_data[4][:, i]  # value predictions
            
            # 归一化GAE（任务特定）
            if USE_TASK_SPECIFIC_NORMALIZATION:
                gae = ppo.normalize_advantages(gae, method='robust')
            else:
                gae = ppo.normalize_advantages(gae, method='standard')

            # 选择训练方法：多步适应 vs 单步训练
            if USE_MULTI_STEP_ADAPTATION:
                losses_history, gradient_norm = ppo.train_multi_step(
                    s, a, gae, reward, v_pred_next, sess, 
                    inner_steps=INNER_LOOP_STEPS, 
                    task_embedding=task_embedding,
                    summary_writer=summary_writer,
                    episode=episode
                )
                # 使用最后一步的损失
                loss_values = losses_history[-1]
            else:
                loss_values, gradient_norm = ppo.train(
                    s, a, gae, reward, v_pred_next, sess,
                    task_embedding=task_embedding,
                    summary_writer=summary_writer,
                    episode=episode
                )
            
            task_loss_all.append(loss_values[1])  # 使用价值损失
            total_gradient_norm += gradient_norm
            
            # 更新统计信息
            statistics.update_task_stats(k, np.mean(reward), loss_values[1], gradient_norm)
        
        # 应用任务权重
        if USE_TASK_WEIGHTS:
            weighted_loss = np.sum(task_loss_all) * task_weights[k]
        else:
            weighted_loss = np.sum(task_loss_all)
        
        loss_batch.append(weighted_loss)

    # 计算episode级别的统计
    episode_loss = np.sum(loss_batch) / len(n_veh)
    episode_reward = np.sum(r_avgs)
    loss_episode.append(episode_loss)
    statistics.update_episode_stats(episode_reward, episode_loss)
    
    episode_time = time.time() - episode_start_time
    
    # 打印训练进度
    if episode % 10 == 0 or episode < 10:
        print(f'Episode {episode:4d}: '
              f'Total Reward: {episode_reward:8.3f}, '
              f'Loss: {episode_loss:8.6f}, '
              f'Gradient Norm: {total_gradient_norm/len(n_veh):6.3f}, '
              f'Time: {episode_time:.2f}s')
        
        # 打印任务级别统计
        for k in range(len(n_veh)):
            stats = statistics.get_task_statistics(k)
            print(f'  Task {k} ({n_veh[k]} UAVs): '
                  f'Reward: {stats["mean_reward"]:6.3f}, '
                  f'Loss: {stats["mean_loss"]:8.6f}, '
                  f'GradNorm: {stats["mean_gradient_norm"]:6.3f}')

    # 检查点保存
    if (episode + 1) % SAVE_CHECKPOINT_EVERY == 0:
        checkpoint_path = f'AC_improved_{optimization_target}_ep{episode+1}_lr{args.lr_meta_a}'
        saved_path = save_models(sess, checkpoint_path, ppo.saver)
        print(f'Checkpoint saved: {saved_path}')

    # 早停检查
    if episode_loss < best_loss:
        best_loss = episode_loss
        no_improvement_count = 0
        # 保存最佳模型
        best_model_path = f'AC_improved_best_{optimization_target}_lr{args.lr_meta_a}'
        save_models(sess, best_model_path, ppo.saver)
    else:
        no_improvement_count += 1

    if early_stopping.should_stop(episode_loss):
        print(f'Early stopping at episode {episode} (no improvement for {EARLY_STOPPING_PATIENCE} episodes)')
        break

# 保存最终模型和结果
final_model_path = f'AC_improved_final_{optimization_target}_{meta_episode}_lr{args.lr_meta_a}'
save_models(sess, final_model_path, ppo.saver)

# 保存训练数据
results_dir = './Train_data'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savetxt(f'{results_dir}/Meta_Reward_improved_{optimization_target}_{meta_episode}_lr{args.lr_meta_a}.txt', record_reward)
np.savetxt(f'{results_dir}/Meta_Loss_improved_{optimization_target}_{meta_episode}_lr{args.lr_meta_a}.txt', loss_episode)

# 保存训练统计
statistics_summary = {
    'n_veh': n_veh,
    'task_weights': task_weights.tolist(),
    'final_episode_rewards': [stats['mean_reward'] for stats in [statistics.get_task_statistics(k) for k in range(len(n_veh))]],
    'final_episode_losses': [stats['mean_loss'] for stats in [statistics.get_task_statistics(k) for k in range(len(n_veh))]],
    'total_episodes': len(record_reward),
    'best_loss': best_loss,
    'configuration': {
        'multi_step_adaptation': USE_MULTI_STEP_ADAPTATION,
        'inner_loop_steps': INNER_LOOP_STEPS,
        'task_weights': USE_TASK_WEIGHTS,
        'task_specific_normalization': USE_TASK_SPECIFIC_NORMALIZATION,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }
}

import json
with open(f'{results_dir}/Meta_Statistics_improved_{optimization_target}_{meta_episode}_lr{args.lr_meta_a}.json', 'w') as f:
    json.dump(statistics_summary, f, indent=2)

summary_writer.close()

print("\n" + "="*80)
print("Meta训练改进版本完成！")
print(f"总训练轮数: {len(record_reward)}")
print(f"最终总奖励: {record_reward[-1]:.3f}")
print(f"最佳损失: {best_loss:.6f}")
print(f"最终模型保存路径: {final_model_path}")
print(f"TensorBoard日志: {log_dir}")

# 任务级别最终统计
print("\n任务级别最终性能:")
for k in range(len(n_veh)):
    stats = statistics.get_task_statistics(k)
    print(f"任务 {k} ({n_veh[k]} UAVs): 平均奖励 = {stats['mean_reward']:.3f}, "
          f"平均损失 = {stats['mean_loss']:.6f}, 权重 = {task_weights[k]:.3f}")

print("="*80)