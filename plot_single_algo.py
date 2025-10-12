import os
import argparse
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_sim_result import load_run


def plot_queue_per_dc(cl: pd.DataFrame, outpath: str):
    dcs = sorted(cl['dc'].unique())
    n_dcs = len(dcs)

    fig, axes = plt.subplots(n_dcs, 1, figsize=(12, 3 * n_dcs), sharex=True)
    if n_dcs == 1:
        axes = [axes]

    for idx, dc in enumerate(dcs):
        dc_data = cl[cl['dc'] == dc].sort_values('time_s')
        axes[idx].plot(dc_data['time_s'], dc_data['q_inf'], label='q_inf', linewidth=1.5)
        axes[idx].plot(dc_data['time_s'], dc_data['q_train'], label='q_train', linewidth=1.5)
        axes[idx].set_ylabel(f'{dc}\nQueue length')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Queue lengths per DC over time', fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_utilization_per_dc(cl: pd.DataFrame, outpath: str):
    cl = cl.copy()
    cl['util'] = cl['busy'] / (cl['busy'] + cl['free'])
    cl['util'] = cl['util'].fillna(0)

    plt.figure(figsize=(12, 6))
    for dc in sorted(cl['dc'].unique()):
        dc_data = cl[cl['dc'] == dc].sort_values('time_s')
        plt.plot(dc_data['time_s'], dc_data['util'], label=dc, alpha=0.7, linewidth=1.5)

    plt.xlabel('Time (s)')
    plt.ylabel('GPU Utilization')
    plt.title('GPU Utilization per DC over time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_job_distribution_by_dc(jb: pd.DataFrame, cl: pd.DataFrame, outpath: str):
    """Show stacked job distribution per DC (training/inference, completed + queued)."""
    # Completed jobs (from job_log)
    dc_type = jb.groupby(['dc', 'type']).size().unstack(fill_value=0)

    # Queued jobs (from cluster_log, take last snapshot)
    last = cl.groupby('dc', as_index=False)[['q_inf', 'q_train']].last()
    last = last.set_index('dc')

    all_dcs = sorted(set(dc_type.index) | set(last.index))
    dc_type = dc_type.reindex(all_dcs, fill_value=0)
    last = last.reindex(all_dcs, fill_value=0)

    # 4 types of jobs
    train_done = dc_type['training'] if 'training' in dc_type.columns else pd.Series(0, index=all_dcs)
    inf_done   = dc_type['inference'] if 'inference' in dc_type.columns else pd.Series(0, index=all_dcs)
    train_q    = last['q_train'] if 'q_train' in last.columns else pd.Series(0, index=all_dcs)
    inf_q      = last['q_inf'] if 'q_inf' in last.columns else pd.Series(0, index=all_dcs)

    # ---- Plot stack ----
    plt.figure(figsize=(9, 5))
    positions = np.arange(len(all_dcs))
    width = 0.6

    p1 = plt.bar(positions, train_done, width, label='Training done', color='#1f77b4')
    p2 = plt.bar(positions, train_q, width, bottom=train_done, label='Training queued', color='#aec7e8')
    p3 = plt.bar(positions, inf_done, width, bottom=train_done + train_q, label='Inference done', color='#ff7f0e')
    p4 = plt.bar(positions, inf_q, width, bottom=train_done + train_q + inf_done, label='Inference queued', color='#ffbb78')

    for i, dc in enumerate(all_dcs):
        base = 0
        for val, color in [
            (train_done[dc], '#1f77b4'),
            (train_q[dc], '#aec7e8'),
            (inf_done[dc], '#ff7f0e'),
            (inf_q[dc], '#ffbb78')
        ]:
            if val > 0:
                plt.text(i, base + val / 2, f"{int(val)}",
                         ha='center', va='center', fontsize=9, color='black')
            base += val

    plt.xticks(positions, all_dcs, rotation=45, ha='right')
    plt.ylabel('Job count')
    plt.title('Job distribution per DC (completed + queued)')
    plt.legend(title='Type', frameon=True)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_job_freq_and_gpus(jb: pd.DataFrame, outpath: str):
    """Plot frequency and n_gpus per job over time (dual y-axis)"""
    jb = jb.copy().sort_values("start_s")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Primary axis: Frequency
    color_freq = 'tab:blue'
    ax1.set_xlabel('Job ID (by start time)')
    ax1.set_ylabel('Frequency (f_used)', color=color_freq)
    line1 = ax1.scatter(range(len(jb)), jb['f_used'],
                        color=color_freq, alpha=0.6, s=20, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color_freq)
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(alpha=0.3)
    # Secondary axis: Number of GPUs
    ax2 = ax1.twinx()
    color_gpu = 'tab:orange'
    ax2.set_ylabel('Number of GPUs (n_gpus)', color=color_gpu)
    line2 = ax2.scatter(range(len(jb)), jb['n_gpus'],
                        color=color_gpu, alpha=0.6, s=20, marker='^', label='n_gpus')
    ax2.tick_params(axis='y', labelcolor=color_gpu)
    ax2.set_ylim(1, 10)
    # Combined legend
    ax1.legend([line1, line2], ['Frequency', 'n_gpus'], loc='upper right')

    plt.title('Job Frequency and GPU Count by Job ID (Completed Jobs)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_jobs_by_ingress(jb: pd.DataFrame, outpath: str):
    """Analyze job arrival patterns by ingress"""
    ingress_counts = jb['ingress'].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count by ingress
    positions = np.arange(len(ingress_counts))
    axes[0].bar(positions, ingress_counts.values, color='coral')
    axes[0].set_title('Jobs per ingress point')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Ingress')
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(ingress_counts.index, rotation=45, ha='right')
    for i, v in enumerate(ingress_counts.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom')

    # Average latency by ingress
    avg_latency = jb.groupby('ingress')['latency_s'].mean().sort_index()
    positions2 = np.arange(len(avg_latency))
    axes[1].bar(positions2, avg_latency.values, color='skyblue')
    axes[1].set_title('Average latency per ingress')
    axes[1].set_ylabel('Latency (s)')
    axes[1].set_xlabel('Ingress')
    axes[1].set_xticks(positions2)
    axes[1].set_xticklabels(avg_latency.index, rotation=45, ha='right')
    for i, v in enumerate(avg_latency.values):
        axes[1].text(i, v, f'{v:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_routing_heatmap(jb: pd.DataFrame, outpath: str):
    """Heatmap showing routing from ingress to DC"""
    routing = pd.crosstab(jb['ingress'], jb['dc'], normalize='index') * 100

    plt.figure(figsize=(14, 8))
    im = plt.imshow(routing.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, label='Percentage of jobs (%)')

    plt.xticks(range(len(routing.columns)), routing.columns, rotation=45, ha='right')
    plt.yticks(range(len(routing.index)), routing.index)
    plt.xlabel('Destination DC')
    plt.ylabel('Ingress Point')
    plt.title('Job Routing Pattern (Ingress → DC)')

    # Add percentage text
    for i in range(len(routing.index)):
        for j in range(len(routing.columns)):
            val = routing.iloc[i, j]
            if val > 0:
                plt.text(j, i, f'{val:.1f}%',
                         ha='center', va='center',
                         color='white' if val > 50 else 'black',
                         fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_busy_per_dc(cl: pd.DataFrame, outpath: str):
    """Plot busy GPUs over time for each DC"""
    dcs = sorted(cl['dc'].unique())
    n_dcs = len(dcs)

    fig, axes = plt.subplots(n_dcs, 1, figsize=(12, 3 * n_dcs), sharex=True)
    if n_dcs == 1:
        axes = [axes]

    for idx, dc in enumerate(dcs):
        dc_data = cl[cl['dc'] == dc].sort_values('time_s')
        axes[idx].plot(dc_data['time_s'], dc_data['busy'], label='Busy GPUs', linewidth=1.5)
        axes[idx].set_ylabel(f'{dc}\nGPU count')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Busy GPUs per DC over time', fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_energy_per_dc(cl: pd.DataFrame, outpath: str):
    """Plot cumulative energy consumption per DC"""
    plt.figure(figsize=(12, 6))

    for dc in sorted(cl['dc'].unique()):
        dc_data = cl[cl['dc'] == dc].sort_values('time_s')
        plt.plot(dc_data['time_s'], dc_data['energy_kJ'], label=dc, alpha=0.7, linewidth=1.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Energy (kJ)')
    plt.title('Cumulative Energy Consumption per DC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Generate per-DC debug plots for cluster/job simulation logs.")
    ap.add_argument("--run", action="append", default=[],
                    help="Khai báo 1 run: NAME=DIR; trong DIR có cluster_log.csv & job_log.csv. "
                         "Ví dụ: baseline=./runs/baseline (có thể dùng nhiều --run)")
    ap.add_argument("--outdir", type=str, default="./debug_figs", help="Thư mục output để lưu các hình.")
    ap.add_argument("--scaledown", type=int, default=1, help="Bước nhảy khi đọc hàng trong log. Dùng khi muốn downsample.")
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("Need at least one --run NAME=DIR")

    os.makedirs(args.outdir, exist_ok=True)

    # Load all runs
    runs_data: Dict[str, dict] = {}

    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Run '{spec}' is invalid. Use NAME=DIR format.")
        name, d = spec.split("=", 1)
        print(f"\nLoading run: {name} from {d}")
        cl, jb = load_run(d, scaledown=args.scaledown)
        runs_data[name] = {'cluster': cl, 'jobs': jb}
        print(f"  Cluster log: {len(cl)} rows, {len(cl['dc'].unique())} DCs")
        print(f"  Job log: {len(jb)} rows")

    for name, data in runs_data.items():
        print(f"Generating debug plots for: {name}")

        cl = data['cluster']
        jb = data['jobs']

        run_outdir = os.path.join(args.outdir, name)
        os.makedirs(run_outdir, exist_ok=True)

        # 1) Queue lengths per DC
        plot_queue_per_dc(cl, os.path.join(run_outdir, "queue_per_dc.png"))

        # 2) Utilization per DC
        plot_utilization_per_dc(cl, os.path.join(run_outdir, "util_per_dc.png"))

        # 3) Busy/Free GPUs per DC
        plot_busy_per_dc(cl, os.path.join(run_outdir, "busy_per_dc.png"))

        # 4) Freq and GPUs by Jobs (jid)
        plot_job_freq_and_gpus(jb, os.path.join(run_outdir, "jid_freq_gpus.png"))

        # 5) Energy per DC
        plot_energy_per_dc(cl, os.path.join(run_outdir, "energy_per_dc.png"))

        # 6) Job distribution by DC
        plot_job_distribution_by_dc(jb, cl, os.path.join(run_outdir, "job_dist.png"))

        # 7) Jobs by ingress
        plot_jobs_by_ingress(jb, os.path.join(run_outdir, "ingress_analysis.png"))

        # 8) Routing heatmap
        plot_routing_heatmap(jb, os.path.join(run_outdir, "routing_heatmap.png"))

    print(f"Debug plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()