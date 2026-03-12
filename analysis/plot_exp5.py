#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 exp5 结果：Meta(area=50/75/100) -> Train(area=25)，n_RB=10, n_veh=6
- TensorBoard 日志: logs/exp5/
- 输出: figures/exp5_convergence_meta{area}.png/pdf, figures/exp5_summary_meta.png/pdf
- 合并: figures/exp5_all.pdf
用法: conda run -n RA_demo python analysis/plot_exp5.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

plt.rcParams['font.family'] = 'Nimbus Roman'
plt.rcParams['axes.unicode_minus'] = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TB_DIR = _PROJECT_ROOT / 'logs' / 'exp5'
_FIGURES_DIR = _PROJECT_ROOT / 'figures'

META_AREA_LIST = [50, 75, 100]
EXP_TAG = "exp5"
CONFIG_ORDER = ['RL', 'FRL', 'MRL', 'MFRL']
STYLE = {
    'RL':   {'color': 'tab:blue',   'ls': '-',  'label': 'MAPPO'},
    'FRL':  {'color': 'tab:orange', 'ls': '-',  'label': 'FedRL'},
    'MRL':  {'color': 'tab:green',  'ls': '--', 'label': 'MRL'},
    'MFRL': {'color': 'tab:red',    'ls': '-.', 'label': 'SA-MFRL (Ours)'},
}


def _config_from_dirname(name):
    if 'MFRL' in name:
        return 'MFRL'
    if 'MRL' in name:
        return 'MRL'
    if 'FRL' in name:
        return 'FRL'
    if 'RL' in name:
        return 'RL'
    return None


def _find_logs_for_meta_area(meta_area):
    if not _TB_DIR.exists():
        return {}
    result = {}
    pat = re.compile(r'_(\d{8}_\d{6})$')
    tag = f'{EXP_TAG}_meta{meta_area}'
    for d in _TB_DIR.iterdir():
        if not d.is_dir():
            continue
        if tag not in d.name:
            continue
        cfg = _config_from_dirname(d.name)
        if cfg is None:
            continue
        m = pat.search(d.name)
        ts = m.group(1) if m else ''
        if cfg not in result or (ts and d.name > result[cfg].name):
            result[cfg] = d
    return result


def load_tb_scalars(log_dir, tag='Train/reward'):
    if not TB_AVAILABLE:
        return None, None
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        if tag not in ea.Tags()['scalars']:
            return None, None
        events = ea.Scalars(tag)
        if not events:
            return None, None
        return np.array([e.value for e in events]), np.array([e.step for e in events])
    except Exception as e:
        print(f"  加载失败 {log_dir}: {e}")
        return None, None


def _plot_convergence(configs, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in CONFIG_ORDER:
        if method not in configs:
            continue
        c = configs[method]
        steps = c['steps'] if c['steps'] is not None else np.arange(len(c['values']))
        s = STYLE.get(method, {'color': 'gray', 'ls': '-', 'label': method})
        ax.plot(steps, c['values'], linewidth=2, color=s['color'], linestyle=s['ls'], label=s['label'])
    ax.set_xlabel('Train Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total SEE', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=False)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=True)
    plt.savefig(Path(save_path).with_suffix('.pdf'), dpi=400, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  OK {save_path}")


def _plot_summary(x_list, data, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in CONFIG_ORDER:
        if method not in data:
            continue
        vals = [data[method].get(x, np.nan) for x in x_list]
        s = STYLE.get(method, {'color': 'gray', 'ls': '-', 'label': method})
        ax.plot(x_list, vals, marker='o', markersize=6, linewidth=2,
                color=s['color'], linestyle=s['ls'], label=s['label'])
    ax.set_xlabel('Meta Area', fontsize=14, fontweight='bold')
    ax.set_ylabel('Final Total SEE (last 10% mean)', fontsize=14, fontweight='bold')
    ax.set_title('Exp5: Meta(area=X) -> Train(area=25), n_RB=10, n_veh=6', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=False)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=True)
    plt.savefig(Path(save_path).with_suffix('.pdf'), dpi=400, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  OK {save_path}")


def _merge_all_pdf():
    pdf_files = []
    for area in META_AREA_LIST:
        p = _FIGURES_DIR / f'exp5_convergence_meta{area}.pdf'
        if p.exists():
            pdf_files.append(p)
    summary_pdf = _FIGURES_DIR / 'exp5_summary_meta.pdf'
    if summary_pdf.exists():
        pdf_files.append(summary_pdf)
    if not pdf_files:
        print("  无 PDF 可合并")
        return
    out_path = _FIGURES_DIR / 'exp5_all.pdf'
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for f in pdf_files:
            writer.append(str(f))
        writer.write(str(out_path))
        print(f"  合并 PDF: {out_path}")
    except ImportError:
        try:
            from PyPDF2 import PdfMerger
            merger = PdfMerger()
            for f in pdf_files:
                merger.append(str(f))
            merger.write(str(out_path))
            merger.close()
            print(f"  合并 PDF: {out_path}")
        except ImportError:
            print("  跳过合并: pip install pypdf")


def main():
    if not TB_AVAILABLE:
        print("请安装: pip install tensorboard")
        return
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = defaultdict(dict)

    print("=" * 80)
    print("绘制 exp5: Meta(area=50/75/100) -> Train(area=25), n_RB=10, n_veh=6")
    print(f"  日志目录: {_TB_DIR}")
    print("=" * 80)

    for area in META_AREA_LIST:
        paths = _find_logs_for_meta_area(area)
        if not paths:
            print(f"  跳过 meta_area={area}: 未找到日志")
            continue
        configs = {}
        for cfg, p in paths.items():
            values, steps = load_tb_scalars(p, 'Train/reward')
            if values is None or len(values) == 0:
                continue
            last_10 = max(1, len(values) // 10)
            final = np.mean(values[-last_10:])
            configs[cfg] = {'values': values, 'steps': steps, 'final': final}
            summary_data[cfg][area] = final
        if len(configs) >= 2:
            _plot_convergence(configs, f'Exp5: Meta(area={area})->Train(area=25), n_veh=6',
                              _FIGURES_DIR / f'exp5_convergence_meta{area}.png')

    if summary_data:
        _plot_summary(META_AREA_LIST, summary_data, _FIGURES_DIR / 'exp5_summary_meta.png')

    _merge_all_pdf()
    print("\n完成")


if __name__ == '__main__':
    main()
