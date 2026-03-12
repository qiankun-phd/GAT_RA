#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 exp5：训练不同 area 的 meta 模型（n_RB=10, n_veh=6）
- area=50, 75, 100，用于验证 Meta(area=X)→Train(area=25) 的迁移规律
- 保存: model/meta_model_AC_SEE_0.2_100_5e-07_nRB10_area{50,75,100}_
用法: conda run -n RA_demo python run_meta_exp5.py
"""

import subprocess
import sys
import os

PYTHON_EXE = "/home/qiankun/.conda/envs/RA_demo/bin/python"

N_RB = 10
N_VEH = 6
META_EPISODE = 100
LR_META_A = 5e-7
LR_META_C = 1e-5
SEED = 1
SIGMA_ADD = 0.2
PATH_LOSS_MODEL = '3GPP_UMa'

META_AREA_LIST = [50, 75, 100]


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    for i, area in enumerate(META_AREA_LIST, 1):
        cmd = [
            PYTHON_EXE, "meta_train_PPO_AC.py",
            "--n_veh_list", str(N_VEH),
            "--n_RB", str(N_RB),
            "--sigma_add", str(SIGMA_ADD),
            "--meta_episode", str(META_EPISODE),
            "--lr_meta_a", str(LR_META_A),
            "--lr_meta_c", str(LR_META_C),
            "--seed", str(SEED),
            "--path_loss_model", PATH_LOSS_MODEL,
            "--area_size", str(area),
        ]
        print("=" * 80)
        print(f"[{i}/{len(META_AREA_LIST)}] 训练 meta: area={area}, n_RB={N_RB}, n_veh={N_VEH}")
        print(f"保存: model/meta_model_AC_SEE_{SIGMA_ADD}_{META_EPISODE}_{LR_META_A}_nRB{N_RB}_area{area}_")
        print("=" * 80)
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            sys.exit(result.returncode)
        print(f"✅ area={area} meta 完成\n")

    print("=" * 80)
    print("exp5 meta 训练完成: area=50, 75, 100")
    print("=" * 80)


if __name__ == "__main__":
    main()
