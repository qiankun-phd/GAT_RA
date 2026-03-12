#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 exp5：Meta(area=50/75/100) → Train(area=25)，n_RB=10, n_veh=6
- 验证不同 meta 面积对迁移到 25×25 的影响
- TensorBoard: logs/exp5/
- 模型: model/exp5/meta{area}/
用法: conda run -n RA_demo python run_exp5_train.py
"""

import subprocess
import sys
import os

PYTHON_EXE = "/home/qiankun/.conda/envs/RA_demo/bin/python"

TRAIN_AREA = 25
N_RB = 10
N_VEH = 6
N_EPISODE = 2000

META_AREA_LIST = [50, 75, 100]
# meta 模型路径格式
def meta_model_path(area):
    return f"meta_model_AC_SEE_0.2_100_5e-07_nRB{N_RB}_area{area}_"

SEED = 2
LR_MAIN = 5e-7
SIGMA_ADD = 0.2
FL_NOISE_SIGMA = 1e-6
TARGET_AVERAGE_STEP = 100
LR_DECAY_AFTER_RATIO = 0.5
LR_DECAY_GAMMA = 0.01
PATH_LOSS_MODEL = '3GPP_UMa'

LOG_DIR = "logs/exp5"

CONFIGS = [
    ("RL", False, False),
    ("FRL", False, True),
    ("MRL", True, False),
    ("MFRL", True, True),
]


def build_cmd(meta_area, model_save_dir, experiment_tag, do_meta, do_fl):
    flags = []
    if do_meta:
        flags.append('--Do_meta')
    if do_fl:
        flags.append('--Do_FL')
    flag_str = ' '.join(flags) if flags else ''
    _lr = f" --lr_decay_after_ratio {LR_DECAY_AFTER_RATIO} --lr_decay_gamma {LR_DECAY_GAMMA}" if LR_DECAY_AFTER_RATIO > 0 else ""
    meta_arg = f" --meta_model_path {meta_model_path(meta_area)}" if do_meta else ""
    base = (
        f"{meta_arg} --area_size {TRAIN_AREA} --n_veh {N_VEH} --n_RB {N_RB} --seed {SEED} "
        f"--sigma_add {SIGMA_ADD} --path_loss_model {PATH_LOSS_MODEL} "
        f"--model_save_dir {model_save_dir} --experiment_tag {experiment_tag} "
        f"--log_dir {LOG_DIR} --no-state_use_episode_progress --fl_noise_sigma {FL_NOISE_SIGMA}{_lr}"
    )
    return f"{PYTHON_EXE} main_PPO_AC.py {flag_str} --use_different_seeds_per_agent --lr_main {LR_MAIN} --n_episode {N_EPISODE} --target_average_step {TARGET_AVERAGE_STEP} {base}".strip().replace('  ', ' ')


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    total = len(META_AREA_LIST) * len(CONFIGS)
    idx = 0

    print("=" * 80)
    print("实验 exp5：Meta(area=50/75/100) → Train(area=25)，n_RB=10, n_veh=6")
    print(f"TensorBoard: {LOG_DIR}/")
    print(f"模型保存: model/exp5/meta{{area}}/")
    print("=" * 80)
    print(f"n_RB={N_RB}, n_veh={N_VEH} (固定)")
    print(f"Meta area: {META_AREA_LIST}")
    print("=" * 80)

    for meta_area in META_AREA_LIST:
        model_save_dir = f"exp5/meta{meta_area}"
        experiment_tag = f"exp5_meta{meta_area}"

        for name, do_meta, do_fl in CONFIGS:
            idx += 1
            cmd = build_cmd(meta_area, model_save_dir, experiment_tag, do_meta, do_fl)
            print(f"\n[{idx}/{total}] meta_area={meta_area} | {name}")
            print(f"  Meta: {meta_model_path(meta_area)}")
            print(f"  模型: model/{model_save_dir}/")
            print("-" * 80)
            try:
                subprocess.run(cmd.split(), check=True, capture_output=False, cwd=cwd)
                print(f"✅ meta_area={meta_area} {name} 完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ meta_area={meta_area} {name} 失败: {e}")
                sys.exit(1)

    print("\n" + "=" * 80)
    print("exp5 训练完成")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR} --port=6008")
    print("=" * 80)


if __name__ == '__main__':
    main()
