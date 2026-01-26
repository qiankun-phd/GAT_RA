#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在多种 FL 配置与 lr_main ∈ [5e-7, 5e-6] 下，找出使 MFRL 优于 MRL 的 FL 组合。
比较维度：收敛速度(conv_ep)、最终收敛值(final)、稳定性(stab_cv)。

用法:
  # 只分析已有 TensorBoard 日志（不跑训练）
  python search_mfrl_beats_mrl.py --mode analyze --use_fixed_lr_grid --n_episode 1000 --fl base,Adapt

  # 跑训练并分析（需先 meta 训练好模型）
  conda activate RA_demo
  python search_mfrl_beats_mrl.py --mode run_and_analyze --use_fixed_lr_grid --n_episode 500 --fl all

  # 固定 lr 网格 [5e-7,1e-6,2e-6,3e-6,5e-6]，lr_main 范围 5e-7~5e-6
  python search_mfrl_beats_mrl.py --mode analyze --use_fixed_lr_grid --n_episode 1000 --fl all --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# 复用 analyze_comparison 的 TensorBoard 解析
try:
    from tensorboard.backend.event_processing import event_accumulator

    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------

LR_MAIN_MIN = 5e-7
LR_MAIN_MAX = 5e-6
# 固定网格：5e-7 ~ 5e-6
LR_GRID_FIXED = [5e-7, 1e-6, 2e-6, 3e-6, 5e-6]

# FL 配置：每个元素 (名称, 命令行额外参数列表)
FL_CONFIGS = [
    ("base", []),  # 仅 Do_FL，无额外
    ("Adapt", ["--fl_adaptive_interval"]),
    ("Soft", ["--fl_soft_aggregation", "--fl_aggregation_weight", "0.7"]),
    ("Layer", ["--fl_layer_wise"]),
    ("Sem", ["--fl_semantic_weighting", "--fl_semantic_temperature", "0.5"]),
    ("Adapt+Soft", ["--fl_adaptive_interval", "--fl_soft_aggregation", "--fl_aggregation_weight", "0.7"]),
    ("Adapt+Layer", ["--fl_adaptive_interval", "--fl_layer_wise"]),
    ("Adapt+Sem", ["--fl_adaptive_interval", "--fl_semantic_weighting", "--fl_semantic_temperature", "0.5"]),
    ("Soft+Layer", ["--fl_soft_aggregation", "--fl_aggregation_weight", "0.7", "--fl_layer_wise"]),
    ("Layer+Sem", ["--fl_layer_wise", "--fl_semantic_weighting", "--fl_semantic_temperature", "0.5"]),
    ("Adapt+Soft+Layer", [
        "--fl_adaptive_interval", "--fl_soft_aggregation", "--fl_aggregation_weight", "0.7", "--fl_layer_wise"
    ]),
    ("Adapt+Soft+Layer+Sem", [
        "--fl_adaptive_interval", "--fl_soft_aggregation", "--fl_aggregation_weight", "0.7",
        "--fl_layer_wise", "--fl_semantic_weighting", "--fl_semantic_temperature", "0.5",
    ]),
]

DEFAULT_N_EPISODE = 500
DEFAULT_N_VEH = 6
DEFAULT_N_RB = 10
DEFAULT_TARGET_AVERAGE_STEP = 100
DEFAULT_META_EPISODE = 100
DEFAULT_SIGMA_ADD = "0.3"
ENV_LABEL = "indoor"
SEMANTIC_AMAX = 1.0
SEMANTIC_BETA = 2.0

/logs/tensorboard/SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10_lr3e-06_FL100_max4_S70_Layer
./logs/tensorboard/SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10_lr3e-06_FL100_max4_Layer_Sem50
./logs/tensorboard/SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10_lr3e-06_FL100_max4_Adapt_S70_Layer_Sem50
./logs/tensorboard/SEE_MAPPO_MFRL_Amax1.0_semB2.0_UAV6_RB10_lr5e-06_FL100_max4

@dataclass
class RunConfig:
    lr_main: float
    fl_name: str | None  # None = MRL
    fl_args: list[str]
    n_episode: int
    n_veh: int
    n_RB: int
    target_average_step: int
    meta_episode: int
    sigma_add: str


def build_log_name(cfg: RunConfig, is_mfrl: bool) -> str:
    """与 main_PPO_AC 一致的 log_name 构造"""
    mode = "MFRL" if is_mfrl else "MRL"
    parts = ["SEE", "MAPPO", mode, f"Amax{SEMANTIC_AMAX}_semB{SEMANTIC_BETA}", f"UAV{cfg.n_veh}_RB{cfg.n_RB}"]
    parts.append(f"lr{cfg.lr_main}")

    if is_mfrl:
        max_fed = max(0, int(0.9 * cfg.n_episode / cfg.target_average_step))
        fl_parts = [f"FL{cfg.target_average_step}_max{max_fed}"]
        args = cfg.fl_args
        if "--fl_adaptive_interval" in args or "Adapt" in cfg.fl_name:
            fl_parts.append("Adapt")
        if "--fl_soft_aggregation" in args:
            w = 0.7
            for i, a in enumerate(args):
                if a == "--fl_aggregation_weight" and i + 1 < len(args):
                    try:
                        w = float(args[i + 1])
                        break
                    except ValueError:
                        pass
            fl_parts.append(f"S{int(w * 100):02d}")
        if "--fl_layer_wise" in args:
            fl_parts.append("Layer")
        if "--fl_semantic_weighting" in args:
            t = 0.5
            for i, a in enumerate(args):
                if a == "--fl_semantic_temperature" and i + 1 < len(args):
                    try:
                        t = float(args[i + 1])
                        break
                    except ValueError:
                        pass
            fl_parts.append(f"Sem{int(t * 100):02d}")
        parts.append("_".join(fl_parts))

    return "_".join(parts)


def log_dir_for(cfg: RunConfig, is_mfrl: bool) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "tensorboard")
    return os.path.join(base, build_log_name(cfg, is_mfrl))


def load_tb_scalars(log_dir: str, tag: str) -> np.ndarray | None:
    if not os.path.isdir(log_dir):
        return None
    events = []
    for f in os.listdir(log_dir):
        if f.startswith("events.out.tfevents"):
            events.append(os.path.join(log_dir, f))
    if not events:
        return None
    latest = max(events, key=os.path.getmtime)
    try:
        ea = event_accumulator.EventAccumulator(latest)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        vals = [e.value for e in ea.Scalars(tag)]
        return np.array(vals) if vals else None
    except Exception:
        return None


def metrics_from_rewards(reward: np.ndarray, tail_ratio: float = 0.1, tail_stability: float = 0.2):
    """从 reward 曲线计算：最终值、收敛速度、稳定性"""
    if reward is None or len(reward) == 0:
        return None
    n = len(reward)
    tail = max(1, int(n * tail_ratio))
    tail_vals = reward[-tail:]

    final = float(np.mean(tail_vals))
    std_tail = float(np.std(tail_vals) + 1e-8)
    mean_tail = float(np.mean(tail_vals)) + 1e-8
    cv = std_tail / abs(mean_tail)

    # 收敛速度：首次达到 90% 最终值的 episode（越小越快）
    thresh = 0.9 * final
    conv_ep = n
    for i in range(n):
        if reward[i] >= thresh:
            conv_ep = i
            break

    # 稳定性：后 20% 的 CV
    stab_tail = max(1, int(n * tail_stability))
    stab_vals = reward[-stab_tail:]
    stab_mean = np.mean(stab_vals) + 1e-8
    stab_std = np.std(stab_vals) + 1e-8
    stab_cv = stab_std / abs(stab_mean)

    return {
        "final": final,
        "conv_ep": conv_ep,
        "cv": cv,
        "stab_cv": stab_cv,
        "mean": float(np.mean(reward)),
        "std": float(np.std(reward)),
    }


def run_main_ppo(cfg: RunConfig, is_mfrl: bool, dry_run: bool = False) -> bool:
    """执行 main_PPO_AC.py 一次。"""
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(root, "main_PPO_AC.py"),
        "--Do_meta",
        "--meta_episode", str(cfg.meta_episode),
        "--n_episode", str(cfg.n_episode),
        "--n_veh", str(cfg.n_veh),
        "--n_RB", str(cfg.n_RB),
        "--lr_main", str(cfg.lr_main),
        "--target_average_step", str(cfg.target_average_step),
        "--sigma_add", cfg.sigma_add,
    ]
    if is_mfrl:
        cmd.append("--Do_FL")
        cmd.extend(cfg.fl_args)

    if dry_run:
        print(" ".join(cmd))
        return True

    try:
        subprocess.run(cmd, cwd=root, check=True, timeout=7200, capture_output=False)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Run failed: {e}")
        return False


def analyze_run(log_dir: str, tag: str = "Train/reward") -> dict | None:
    arr = load_tb_scalars(log_dir, tag)
    return metrics_from_rewards(arr)


def _win_to_json(w: dict) -> dict:
    out = {
        "lr": float(w["lr"]),
        "fl": w["fl"],
        "faster": bool(w["faster"]),
        "higher": bool(w["higher"]),
        "stabler": bool(w["stabler"]),
        "mrl_final": float(w["mrl"]["final"]),
        "mrl_conv_ep": int(w["mrl"]["conv_ep"]),
        "mrl_stab_cv": float(w["mrl"]["stab_cv"]),
        "mfrl_final": float(w["mfrl"]["final"]),
        "mfrl_conv_ep": int(w["mfrl"]["conv_ep"]),
        "mfrl_stab_cv": float(w["mfrl"]["stab_cv"]),
    }
    return out


def run_search(
    lr_list: list[float],
    fl_list: list[tuple[str, list[str]]],
    n_episode: int,
    n_veh: int,
    n_RB: int,
    target_average_step: int,
    meta_episode: int,
    sigma_add: str,
    mode: str,
    dry_run: bool,
    out_path: str | None = None,
) -> list[dict]:
    assert mode in ("run", "analyze", "run_and_analyze")

    results_mrl: dict[float, dict] = {}
    results_mfrl: dict[tuple[float, str], dict] = {}

    configs_mrl = [
        RunConfig(
            lr_main=lr,
            fl_name=None,
            fl_args=[],
            n_episode=n_episode,
            n_veh=n_veh,
            n_RB=n_RB,
            target_average_step=target_average_step,
            meta_episode=meta_episode,
            sigma_add=sigma_add,
        )
        for lr in lr_list
    ]

    for cfg in configs_mrl:
        if mode in ("run", "run_and_analyze"):
            run_main_ppo(cfg, is_mfrl=False, dry_run=dry_run)
        if mode in ("analyze", "run_and_analyze") and not dry_run:
            ld = log_dir_for(cfg, is_mfrl=False)
            m = analyze_run(ld)
            if m:
                results_mrl[cfg.lr_main] = m

    for lr in lr_list:
        for fl_name, fl_args in fl_list:
            cfg = RunConfig(
                lr_main=lr,
                fl_name=fl_name,
                fl_args=fl_args,
                n_episode=n_episode,
                n_veh=n_veh,
                n_RB=n_RB,
                target_average_step=target_average_step,
                meta_episode=meta_episode,
                sigma_add=sigma_add,
            )
            if mode in ("run", "run_and_analyze"):
                run_main_ppo(cfg, is_mfrl=True, dry_run=dry_run)
            if mode in ("analyze", "run_and_analyze") and not dry_run:
                ld = log_dir_for(cfg, is_mfrl=True)
                m = analyze_run(ld)
                if m:
                    results_mfrl[(lr, fl_name)] = m

    if dry_run or mode == "run":
        return []

    # 比较：找出 MFRL > MRL 的 (lr, FL)
    print("\n" + "=" * 80)
    print("MFRL 优于 MRL 的 (lr_main, FL) 组合")
    print("比较维度：收敛速度(conv_ep 更小)、最终值(final 更大)、稳定性(stab_cv 更小)")
    print("=" * 80)

    wins = []
    for lr in lr_list:
        mrl = results_mrl.get(lr)
        if not mrl:
            continue
        for fl_name, _ in fl_list:
            mfrl = results_mfrl.get((lr, fl_name))
            if not mfrl:
                continue
            faster = mfrl["conv_ep"] < mrl["conv_ep"]
            higher = mfrl["final"] > mrl["final"]
            stabler = mfrl["stab_cv"] < mrl["stab_cv"]
            if faster or higher or stabler:
                wins.append({
                    "lr": lr,
                    "fl": fl_name,
                    "faster": faster,
                    "higher": higher,
                    "stabler": stabler,
                    "mrl": mrl,
                    "mfrl": mfrl,
                })

    if not wins:
        print("未找到 MFRL 优于 MRL 的配置。")
        return []

    for w in wins:
        lr, fl = w["lr"], w["fl"]
        mr, mf = w["mrl"], w["mfrl"]
        s = []
        if w["faster"]:
            s.append(f"收敛更快(conv_ep {mf['conv_ep']} < {mr['conv_ep']})")
        if w["higher"]:
            s.append(f"最终更高(final {mf['final']:.4f} > {mr['final']:.4f})")
        if w["stabler"]:
            s.append(f"更稳定(stab_cv {mf['stab_cv']:.4f} < {mr['stab_cv']:.4f})")
        print(f"  lr_main={lr:.0e}  FL={fl}  =>  {'; '.join(s)}")

    print("\n" + "-" * 80)
    print("汇总：按 FL 统计出现次数（MFRL 至少在一维优于 MRL）")
    fl_count: dict[str, int] = defaultdict(int)
    for w in wins:
        fl_count[w["fl"]] += 1
    for fl, c in sorted(fl_count.items(), key=lambda x: -x[1]):
        print(f"  {fl}: {c} 次")

    if out_path:
        data = {"wins": [_win_to_json(w) for w in wins], "fl_count": dict(fl_count)}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")

    return wins


def main():
    ap = argparse.ArgumentParser(description="搜索使 MFRL 优于 MRL 的 FL 配置")
    ap.add_argument("--mode", choices=["run", "analyze", "run_and_analyze"], default="run_and_analyze",
                    help="run=只跑实验, analyze=只分析已有日志, run_and_analyze=跑并分析")
    ap.add_argument("--lr_min", type=float, default=LR_MAIN_MIN)
    ap.add_argument("--lr_max", type=float, default=LR_MAIN_MAX)
    ap.add_argument("--lr_num", type=int, default=5, help="lr 在 [lr_min,lr_max] 对数均匀取点数")
    ap.add_argument("--n_episode", type=int, default=DEFAULT_N_EPISODE)
    ap.add_argument("--n_veh", type=int, default=DEFAULT_N_VEH)
    ap.add_argument("--n_RB", type=int, default=DEFAULT_N_RB)
    ap.add_argument("--target_average_step", type=int, default=DEFAULT_TARGET_AVERAGE_STEP)
    ap.add_argument("--meta_episode", type=int, default=DEFAULT_META_EPISODE)
    ap.add_argument("--sigma_add", type=str, default=DEFAULT_SIGMA_ADD)
    ap.add_argument("--fl", type=str, default="all",
                    help="all | base,Adapt,Soft,Layer,Sem 等逗号分隔")
    ap.add_argument("--use_fixed_lr_grid", action="store_true",
                    help="使用固定 lr 网格 [5e-7,1e-6,2e-6,3e-6,5e-6]")
    ap.add_argument("--out", type=str, default=None, help="将结果保存为 JSON 文件")
    ap.add_argument("--dry_run", action="store_true", help="只打印命令不执行")

    args = ap.parse_args()

    if not TB_AVAILABLE:
        print("需要 tensorboard。pip install tensorboard")
        sys.exit(1)

    if args.use_fixed_lr_grid:
        lr_list = LR_GRID_FIXED.copy()
    else:
        lr_list = np.linspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_num)
        lr_list = [float(10 ** x) for x in lr_list]

    if args.fl == "all":
        fl_list = FL_CONFIGS
    else:
        names = [x.strip() for x in args.fl.split(",")]
        fl_list = [(n, []) for n in names]
        for spec in FL_CONFIGS:
            if spec[0] in names:
                idx = next(i for i, (a, _) in enumerate(fl_list) if a == spec[0])
                fl_list[idx] = (spec[0], spec[1])

    run_search(
        lr_list=lr_list,
        fl_list=fl_list,
        n_episode=args.n_episode,
        n_veh=args.n_veh,
        n_RB=args.n_RB,
        target_average_step=args.target_average_step,
        meta_episode=args.meta_episode,
        sigma_add=args.sigma_add,
        mode=args.mode,
        dry_run=args.dry_run,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
