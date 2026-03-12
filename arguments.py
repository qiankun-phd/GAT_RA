import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--gpu_number',
        default="1",
        help='gpu usage')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed')
    parser.add_argument(
        '--set_random_seed',
        type=int,
        default=2,
        help='set_random_seed')
    parser.add_argument(
        '--use_different_seeds_per_agent',
        action='store_true',
        default=False,
        help='use different random seeds for each agent to ensure parameter diversity (default: False, all agents use the same seed)')
    parser.add_argument(
        '--meta_episode',
        type=int,
        default=100,
        help='meta episodes (default: 100)')
    parser.add_argument(
        '--n_episode',
        type=int,
        default=1000,
        help='main PPO times (default: 1000)')
    parser.add_argument(
        '--n_test_episode',
        type=int,
        default=100,
        help='main test times (default: 100)')
    parser.add_argument(
        '--IS_PPO',
        action='store_true',
        default=True,
        help='Is policy gradient (default: False, set to True for PPO)')
    parser.add_argument(
        '--n_veh',
        type=int,
        default = 6,
        help='number of vehicles (default: 6)')
    parser.add_argument(
        '--n_veh_list',
        type=lambda s: [int(item) for item in s.strip('[]').split(',')],
        default=[2,4,8],
        help='number of vehicles (for different tasks), e.g., [2,4,6,8,10] or 2,4,6,8,10')
    parser.add_argument(
        '--n_RB',
        type=int,
        default=10,
        help='number of resource blocks')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='discount rate gamma (default: 0.9)')
    parser.add_argument(
        '--lambda_advantage',
        type=float,
        default=0.98,
        help='discount rate for GAE (default: 1)')
    parser.add_argument(
        '--meta_gamma',
        type=float,
        default=0.9,
        help='discount rate gamma (default: 0.9)')
    parser.add_argument(
        '--meta_batch_size',
        type=int,
        default=512,
        help='meta batch size (default: 512)')
    parser.add_argument(
        '--RB_action_bound',
        type=float,
        default=1,
        help='RB action bound (default: 1)')
    parser.add_argument(
        '--weight_for_L_vf',
        type=float,
        default=0.5,
        help='loss weight for tf.reduce_mean(tf.square(self.discounted_r - self.v))')
    parser.add_argument(
        '--weight_for_entropy',
        type=float,
        default=0.01,
        help='loss weight for entropy (default: 0.01)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.5,
        help='PPO clip ratio epsilon, clip range [1-eps, 1+eps] (default 0.5)')
    parser.add_argument(
        '--lr_meta_a', type=float, default=5e-7, help='learning rate for actor (default: 5e-7)')
    parser.add_argument(
        '--lr_meta_c', type=float, default=1e-5, help='learning rate for critic (default: 1e-5)')
    parser.add_argument(
        '--lr_main', type=float, default=1e-6, help='learning rate for PPO (default: 1e-6)')
    parser.add_argument(
        '--lr_decay_after_ratio', type=float, default=0.0,
        help='after this ratio of n_episode, lr linearly decays to lr_main*lr_decay_gamma (0=no decay, e.g. 0.5=from 50%%)')
    parser.add_argument(
        '--lr_decay_gamma', type=float, default=0.5,
        help='linear decay end: lr = lr_main*gamma at last episode (default: 0.5)')
    parser.add_argument(
        '--minibatch_steps',
        type=int,
        default=32,
        help='minibatch_steps ppo (default: 32)')
    # SAC 专用参数（main_SAC_AC.py）
    parser.add_argument('--lr_sac_actor', type=float, default=3e-4, help='SAC actor lr (default: 3e-4)')
    parser.add_argument('--lr_sac_critic', type=float, default=3e-4, help='SAC critic lr (default: 3e-4)')
    parser.add_argument('--sac_alpha', type=float, default=0.2, help='SAC entropy coef (default: 0.2)')
    parser.add_argument('--sac_tau', type=float, default=0.005, help='SAC target soft update (default: 0.005)')
    parser.add_argument('--sac_buffer_size', type=int, default=100000, help='SAC replay buffer size (default: 100000)')
    parser.add_argument('--sac_batch_size', type=int, default=256, help='SAC batch size (default: 256)')
    parser.add_argument('--sac_warmup_steps', type=int, default=1000, help='SAC warmup steps before train (default: 1000)')
    parser.add_argument(
        '--save_path',
        default='meta_model_',
        help='directory to save models (default: meta_model_)')
    parser.add_argument(
        '--meta_model_path',
        type=str,
        default='',
        help='Override meta model path when loading. Empty=auto from save_path+AC_SEE+sigma_add+meta_episode+lr+area (default: )')
    parser.add_argument(
        '--target_average_step',
        type=int,
        default=100,
        help='target_average_step (default: 100)')
    parser.add_argument(
        '--save_best',
        action='store_true',
        default=True,
        help='save best model by rolling reward during training (default: True)')
    parser.add_argument(
        '--no_save_best',
        action='store_true',
        default=False,
        help='disable save best model (default: False)')
    parser.add_argument(
        '--save_best_rolling',
        type=int,
        default=50,
        help='rolling window episodes for best reward (default: 50)')
    parser.add_argument(
        '--Do_meta',
        action='store_true',
        default=False,
        help='if use meta learning to initialize (default: False)')
    parser.add_argument(
        '--Do_FL',
        action='store_true',
        default=False,
        help='if use federated learning (model averaging) (default: False)')
    parser.add_argument(
        '--fl_adaptive_interval',
        action='store_true',
        default=False,
        help='if use adaptive aggregation interval (stage-based training) (default: False)')
    parser.add_argument(
        '--fl_soft_aggregation',
        action='store_true',
        default=False,
        help='if use soft aggregation (partial replacement) instead of hard replacement (default: False)')
    parser.add_argument(
        '--fl_aggregation_weight',
        type=float,
        default=0.7,
        help='aggregation weight for soft aggregation (0.0-1.0, 1.0=hard replacement, 0.7=soft aggregation) (default: 0.7, only used when --fl_soft_aggregation is enabled)')
    parser.add_argument(
        '--fl_layer_wise',
        action='store_true',
        default=False,
        help='if use layer-wise federated aggregation (only aggregate encoder layers, keep decision heads personalized) (default: False)')
    parser.add_argument(
        '--fl_semantic_weighting',
        action='store_true',
        default=False,
        help='if use semantic-aware weighting based on Semantic Energy Efficiency (SEE) instead of success rate (default: False)')
    parser.add_argument(
        '--fl_semantic_temperature',
        type=float,
        default=0.5,
        help='temperature coefficient for semantic weighting softmax (smaller=more emphasis on good agents) (default: 0.5)')
    parser.add_argument(
        '--fl_noise_sigma',
        type=float,
        default=1e-8,
        help='std of Gaussian noise used to initialize FL aggregation accumulators (backup-style noisy mean init, default: 1e-8)')
    parser.add_argument(
        '--fl_use_success_rate_weighting',
        action='store_true',
        default=False,
        help='if use success rate as aggregation weights instead of uniform weights (default: False, use uniform weights - standard FedAvg)')
    parser.add_argument(
        '--state_use_episode_progress',
        action='store_true',
        default=True,
        help='if include episode progress (ind_episode/n_episode) in state vector (default: True). Use --no-state_use_episode_progress to disable it for consistency across different n_episode values')
    parser.add_argument(
        '--no-state_use_episode_progress',
        dest='state_use_episode_progress',
        action='store_false',
        help='disable episode progress in state vector to ensure consistency across different n_episode values (use this when comparing experiments with different n_episode)')
    parser.add_argument(
        '--fl_average_actor',
        action='store_true',
        default=False,
        help='if average Actor network in FL (default: False)')
    parser.add_argument(
        '--fl_average_critic1',
        action='store_true',
        default=False,
        help='if average Critic1 network in FL (default: False, use --fl_average_critic1 to enable)')
    parser.add_argument(
        '--fl_average_critic2',
        action='store_true',
        default=False,
        help='if average Critic2 network in FL (default: False, use --fl_average_critic2 to enable)')
    parser.add_argument(
        '--fl_average_all_critics',
        action='store_true',
        default=False,
        help='if average all Critic networks (equivalent to --fl_average_critic1 --fl_average_critic2)')
    parser.add_argument(
        '--same_env',
        action='store_true',
        default=False,
        help='if use meta learning to initialize (default: True)')
    parser.add_argument(
        '--n_hidden_1',
        type=int,
        default=512,
        help='n_hidden_1 (default: 512)')
    parser.add_argument(
        '--n_hidden_2',
        type=int,
        default=256,
        help='n_hidden_2 (default: 256)')
    parser.add_argument(
        '--n_hidden_3',
        type=int,
        default=128,
        help='n_hidden_3 (default: 128)')

    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--sigma_add',
        type=float,
        default=0.3,
        help='sigma add on policy Normal scale for exploration (default: 0.1)')
    parser.add_argument(
        '--sigma_max',
        type=float,
        default=2.0,
        help='SAC only: clip policy Normal scale to [sigma_add+1e-5, sigma_max] to avoid instability (default: 2.0)')
    parser.add_argument(
        '--env_choice',
        type=int,
        default=0,
        help='0: indoor, 1: cannon, 2: urban, 3: rural')
    # optimization_target已固定为SE_EE，不再提供选择
    # parser.add_argument(
    #     '--optimization_target',
    #     type=str,
    #     default='SE_EE',
    #     choices=['SE', 'EE', 'SE_EE'],
    #     help='Optimization target: SE (Spectral Efficiency only), EE (Energy Efficiency only), or SE_EE (weighted combination, default)')
    parser.add_argument(
        '--beta',
        type=float,
        default=0.5,
        help='Weight for SE in reward calculation (0.0-1.0), Semantic-EE weight = 1 - beta (default: 0.5). Reward = beta * SE + (1-beta) * Semantic-EE')
    parser.add_argument(
        '--circuit_power',
        type=float,
        default=0.06,
        help='Circuit power in linear scale for EE calculation (default: 0.06)')
    parser.add_argument(
        '--sig2_dB',
        type=float,
        default=-160,
        help='Noise power spectral density in dB (default: -160). Thermal -204; use -60~-65 for SINR -10~20 dB')
    parser.add_argument(
        '--area_size',
        type=float,
        default=25.0,
        help='Scenario area side length in m (default: 25). GBS at center.')
    parser.add_argument(
        '--model_save_dir',
        type=str,
        default='',
        help='Subdir under model/ for saving (e.g. exp1_area25). Empty=default model/')
    parser.add_argument(
        '--experiment_tag',
        type=str,
        default='',
        help='Tag for tensorboard run name (e.g. exp1_area25). Empty=no tag')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/tensorboard',
        help='Base directory for tensorboard logs (e.g. logs/exp4). Default: logs/tensorboard')
    parser.add_argument(
        '--path_loss_offset_dB',
        type=float,
        default=0,
        help='Offset subtracted from path loss (dB). Use 55 to raise min SNR to >= -20 dB for area=500 (default: 0)')
    parser.add_argument(
        '--path_loss_model',
        type=str,
        default='A2G',
        choices=['A2G', '3GPP_UMa'],
        help='Path loss model: A2G (ITU) or 3GPP_UMa (PL=128.1+37.6*log10(d_3d/1000)) (default: A2G)')
    parser.add_argument(
        '--semantic_A_max',
        type=float,
        default=1.0,
        help='Maximum semantic accuracy (mAP) for semantic communication (default: 1.0, deprecated)')
    parser.add_argument(
        '--semantic_beta',
        type=float,
        default=2.0,
        help='Compression ratio sensitivity parameter for semantic accuracy (default: 2.0)')
    parser.add_argument(
        '--semantic_A1',
        type=float,
        default=1.0,
        help='Upper bound of semantic accuracy in Sigmoid model (default: 1.0)')
    parser.add_argument(
        '--semantic_A2',
        type=float,
        default=0.2,
        help='Lower bound of semantic accuracy in Sigmoid model (default: 0.2)')
    parser.add_argument(
        '--semantic_C1',
        type=float,
        default=5.0,
        help='Slope parameter in Sigmoid model (default: 5.0)')
    parser.add_argument(
        '--semantic_C2',
        type=float,
        default=2.0,
        help='Offset parameter in Sigmoid model (default: 2.0)')
    # Task similarity model: Q = A_peak*(1-e^{-xi*rho})/(1+e^{-zeta*(gamma-gamma0)}) + b (SNR -20~60 dB)
    parser.add_argument('--task_sim_A_peak', type=float, default=0.7128, help='Task sim model A_peak (default: 0.7128)')
    parser.add_argument('--task_sim_xi', type=float, default=10.0, help='Task sim model xi (default: 10.0)')
    parser.add_argument('--task_sim_zeta', type=float, default=0.2313, help='Task sim model zeta (default: 0.2313)')
    parser.add_argument('--task_sim_gamma0', type=float, default=0.0, help='Task sim model gamma0 (default: 0.0)')
    parser.add_argument('--task_sim_b', type=float, default=0.3249, help='Task sim model b (default: 0.3249)')
    parser.add_argument(
        '--collision_penalty',
        type=float,
        default=-0.1,
        help='Penalty for RB collision (scaled down for better reward balance, default: -0.1)')
    parser.add_argument(
        '--low_accuracy_penalty',
        type=float,
        default=-0.05,
        help='Penalty for low semantic accuracy (scaled down for better reward balance, default: -0.05)')
    parser.add_argument(
        '--accuracy_threshold',
        type=float,
        default=0.5,
        help='Minimum acceptable semantic accuracy threshold (default: 0.5)')
    args = parser.parse_args()

    # args.cuda = not args.no_cuda and tf.test.is_gpu_available()

    return args