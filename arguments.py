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
        type=list,
        default=[2, 4, 8],
        help='number of vehicles (for different tasks)')
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
        help='ratio clip value (default 0.5)')
    parser.add_argument(
        '--lr_meta_a', type=float, default=5e-7, help='learning rate for actor (default: 5e-7)')
    parser.add_argument(
        '--lr_meta_c', type=float, default=1e-5, help='learning rate for critic (default: 1e-5)')
    parser.add_argument(
        '--lr_main', type=float, default=1e-6, help='learning rate for PPO (default: 1e-6)')
    parser.add_argument(
        '--minibatch_steps',
        type=int,
        default=32,
        help='minibatch_steps ppo (default: 32)')
    parser.add_argument(
        '--save_path',
        default='meta_model_',
        help='directory to save models (default: meta_model_)')
    parser.add_argument(
        '--target_average_step',
        type=int,
        default=100,
        help='target_average_step (default: 100)')
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
        help='sigma add on (default: 0.3)')
    parser.add_argument(
        '--env_choice',
        type=int,
        default=0,
        help='0: indoor, 1: cannon, 2: urban, 3: rural')
    parser.add_argument(
        '--optimization_target',
        type=str,
        default='SE_EE',
        choices=['SE', 'EE', 'SE_EE'],
        help='Optimization target: SE (Spectral Efficiency only), EE (Energy Efficiency only), or SE_EE (weighted combination, default)')
    parser.add_argument(
        '--beta',
        type=float,
        default=0.5,
        help='Weight for SE in reward calculation when optimization_target=SE_EE (0.0-1.0), EE weight = 1 - beta (default: 0.5)')
    parser.add_argument(
        '--circuit_power',
        type=float,
        default=0.06,
        help='Circuit power in linear scale for EE calculation (default: 0.06)')
    parser.add_argument(
        '--semantic_A_max',
        type=float,
        default=1.0,
        help='Maximum semantic accuracy (mAP) for semantic communication (default: 1.0)')
    parser.add_argument(
        '--semantic_beta',
        type=float,
        default=2.0,
        help='Compression ratio sensitivity parameter for semantic accuracy (default: 2.0)')
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
    parser.add_argument(
        '--use_gat',
        action='store_true',
        default=False,
        help='Use Graph Attention Network instead of MLP (default: False, set --use_gat to enable)')
    parser.add_argument(
        '--num_gat_heads',
        type=int,
        default=4,
        help='Number of attention heads in GAT (default: 4)')
    args = parser.parse_args()

    # args.cuda = not args.no_cuda and tf.test.is_gpu_available()

    return args