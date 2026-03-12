#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 exp6：Meta(area=100) → Train(area=25)，n_RB=16，n_veh=2,4,6,8,10
- 使用 meta_model_AC_SEE_0.2_100_5e-07_area100_（原 n_RB=10 的 meta）
- 训练时 n_RB=16，n_veh 扫描（2,4,6,8,10）
- TensorBoard: logs/exp6/
- 模型: model/exp6/nveh{n}/
用法: conda run -n RA_demo python run_exp6_train.py
"""

import subprocess
import sys
import os

PYTHON_EXE = "/home/qiankun/.conda/envs/RA_demo/bin/python"

TRAIN_AREA = 25
N_RB = 16  # 训练时固定 n_RB=16
META_MODEL_PATH = "meta_model_AC_SEE_0.2_100_5e-07_area100_"
N_EPISODE = 3000
N_VEH_LIST = [2, 4, 6, 8, 10]

SEED = 2
LR_MAIN = 5e-7
SIGMA_ADD = 0.2
FL_NOISE_SIGMA = 1e-6
TARGET_AVERAGE_STEP = 100
LR_DECAY_AFTER_RATIO = 0.5
LR_DECAY_GAMMA = 0.01
PATH_LOSS_MODEL = '3GPP_UMa'

LOG_DIR = "logs/exp6"

CONFIGS = [
    ("RL", False, False),
    ("FRL", False, True),
    ("MRL", True, False),
    ("MFRL", True, True),
]


def build_cmd(n_veh, model_save_dir, experiment_tag, do_meta, do_fl):
    flags = []
    if do_meta:
        flags.append('--Do_meta')
    if do_fl:
        flags.append('--Do_FL')
    flag_str = ' '.join(flags) if flags else ''
    _lr = f" --lr_decay_after_ratio {LR_DECAY_AFTER_RATIO} --lr_decay_gamma {LR_DECAY_GAMMA}" if LR_DECAY_AFTER_RATIO > 0 else ""
    meta_arg = f" --meta_model_path {META_MODEL_PATH}" if do_meta else ""
    base = (
        f"{meta_arg} --area_size {TRAIN_AREA} --n_veh {n_veh} --n_RB {N_RB} --seed {SEED} "
        f"--sigma_add {SIGMA_ADD} --path_loss_model {PATH_LOSS_MODEL} "
        f"--model_save_dir {model_save_dir} --experiment_tag {experiment_tag} "
        f"--log_dir {LOG_DIR} --no-state_use_episode_progress --fl_noise_sigma {FL_NOISE_SIGMA}{_lr}"
    )
    return f"{PYTHON_EXE} main_PPO_AC.py {flag_str} --use_different_seeds_per_agent --lr_main {LR_MAIN} --n_episode {N_EPISODE} --target_average_step {TARGET_AVERAGE_STEP} {base}".strip().replace('  ', ' ')


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    total = len(N_VEH_LIST) * len(CONFIGS)
    idx = 0

    print("=" * 80)
    print("实验 exp6：Meta(area=100) → Train(area=25)，n_RB=16，n_veh=2,4,6,8,10")
    print(f"Meta: {META_MODEL_PATH}")
    print(f"TensorBoard: {LOG_DIR}/")
    print(f"模型保存: model/exp6/nveh{{n}}/")
    print("=" * 80)
    print(f"n_veh: {N_VEH_LIST}, n_RB={N_RB} (固定)")
    print("=" * 80)

    for n_veh in N_VEH_LIST:
        model_save_dir = f"exp6/nveh{n_veh}"
        experiment_tag = f"exp6_nveh{n_veh}"

        for name, do_meta, do_fl in CONFIGS:
            idx += 1
            cmd = build_cmd(n_veh, model_save_dir, experiment_tag, do_meta, do_fl)
            print(f"\n[{idx}/{total}] n_veh={n_veh} | {name}")
            print(f"  模型: model/{model_save_dir}/")
            print("-" * 80)
            try:
                subprocess.run(cmd.split(), check=True, capture_output=False, cwd=cwd)
                print(f"✅ n_veh={n_veh} {name} 完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ n_veh={n_veh} {name} 失败: {e}")
                sys.exit(1)

    print("\n" + "=" * 80)
    print("exp6 训练完成")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR} --port=6008")
    print("=" * 80)


if __name__ == '__main__':
    main()
