import os
import argparse
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_run(dir_path: str, scaledown: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cluster_log.csv and job_log.csv from dir_path."""
    cl_path = os.path.join(dir_path, "cluster_log.csv")
    jb_path = os.path.join(dir_path, "job_log.csv")
    if not os.path.isfile(cl_path):
        raise FileNotFoundError(f"Missing cluster_log.csv in {dir_path}")
    if not os.path.isfile(jb_path):
        raise FileNotFoundError(f"Missing job_log.csv in {dir_path}")
    cl = pd.read_csv(cl_path)
    jb = pd.read_csv(jb_path)

    if scaledown > 1:
        cl = cl.iloc[::scaledown, :].reset_index(drop=True)
        jb = jb.iloc[::scaledown, :].reset_index(drop=True)

    return cl, jb


def aggregate_cluster(cl: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cluster log to system-level by time.
       - total_power_W: sum power_W across DC
       - total_energy_kJ: sum energy_kJ across DC (system cumulative)
       - total_job_unit: sum job_size ynit accross DC
       - util: (sum busy) / (sum (busy+free))
       - q_inf / q_train: sums across DC
       - freq_avg: simple average of freq across DC (for reference only)
    """
    df = cl.copy()
    # Ensure column types
    for col in ["time_s", "power_W", "energy_kJ", "acc_job_unit", "busy", "free", "q_inf", "q_train", "freq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_gpus"] = df["busy"] + df["free"]
    g = df.groupby("time_s", as_index=False).agg(
        total_power_W=("power_W", "sum"),
        total_energy_kJ=("energy_kJ", "sum"),
        total_job_unit=("acc_job_unit", "sum"),
        busy_sum=("busy", "sum"),
        total_gpus_sum=("total_gpus", "sum"),
        q_inf_sum=("q_inf", "sum"),
        q_train_sum=("q_train", "sum"),
        freq_avg=("freq", "mean")
    )
    g["util"] = np.where(g["total_gpus_sum"] > 0, g["busy_sum"] / g["total_gpus_sum"], 0.0)
    return g


def plot_lines_over_time(series_dict: Dict[str, pd.DataFrame], x, y, ylabel, title, outpath):
    plt.figure()
    for name, df in series_dict.items():
        if x in df.columns and y in df.columns:
            plt.plot(df[x], df[y], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_queues_over_time(series_dict: Dict[str, pd.DataFrame], outpath, has_infer=True):
    # Two subplots would violate the "one chart per figure" rule in this environment.
    # So we produce a single figure with two overlaid lines per run for q_inf and q_train.
    plt.figure()
    for name, df in series_dict.items():
        if {"time_s", "q_inf_sum", "q_train_sum"}.issubset(df.columns):
            if has_infer:
                plt.plot(df["time_s"], df["q_inf_sum"], label=f"{name}-q_inf")
            plt.plot(df["time_s"], df["q_train_sum"], label=f"{name}-q_train")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue length (requests)")
    plt.title("Queue lengths over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_latency_hist(job_dict: Dict[str, pd.DataFrame], job_type: str, outpath: str, bins: int = 40):
    plt.figure()
    for name, df in job_dict.items():
        if {"type", "latency_s"}.issubset(df.columns):
            d = df[df["type"] == job_type]
            if len(d) > 0:
                plt.hist(d["latency_s"], bins=bins, alpha=0.5, label=name, density=False)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title(f"Latency histogram — {job_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_energy_vs_latency(job_dict: Dict[str, pd.DataFrame], outpath: str, sample: int = 5000):
    plt.figure()
    for name, df in job_dict.items():
        if {"size", "E_pred", "latency_s"}.issubset(df.columns):
            d = df.copy()
            d["E_job_pred_J"] = d["size"] * d["E_pred"]
            if len(d) > sample:
                d = d.sample(sample, random_state=0)
            plt.scatter(d["latency_s"], d["E_job_pred_J"], s=10, label=name)
    plt.xlabel("Latency (s)")
    plt.ylabel("Predicted energy per job (J)")
    plt.title("Energy vs Latency (predicted)")
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_total_energy_bar(agg_dict: Dict[str, pd.DataFrame], outpath: str):
    names, totals = [], []
    for name, df in agg_dict.items():
        if "total_energy_kJ" in df.columns and len(df) > 0:
            names.append(name)
            totals.append(float(df["total_energy_kJ"].iloc[-1]))
    plt.figure()
    positions = np.arange(len(names))
    plt.bar(positions, totals)

    for pos, total in zip(positions, totals):
        plt.text(
            pos, total, f"{total:.1f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.xticks(positions, names, rotation=20)
    plt.ylabel("Total energy (kJ)")
    plt.title("Final total energy per run")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_throughput(job_dict: Dict[str, pd.DataFrame], outpath: str, bin_size_s: float = 5.0):
    plt.figure()
    for name, df in job_dict.items():
        if "finish_s" in df.columns:
            t = pd.to_numeric(df["finish_s"], errors="coerce")
            t = t.dropna()
            if len(t) == 0:
                continue
            # Bin to per-interval completions (jobs/s)
            bins = np.arange(t.min(), t.max() + bin_size_s, bin_size_s)
            counts, edges = np.histogram(t, bins=bins)
            throughput = counts / bin_size_s
            centers = (edges[:-1] + edges[1:]) / 2.0
            plt.plot(centers, throughput, label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (jobs/s)")
    plt.title(f"Throughput vs time (bin={bin_size_s:.0f}s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_energy_by_load(agg_dict: Dict[str, pd.DataFrame], outpath: str):
    names, total_energy_kJ, total_load = [], [], []
    for name, df in agg_dict.items():
        if "total_energy_kJ" in df.columns and "total_job_unit" in df.columns and len(df) > 0:
            names.append(name)
            total_energy_kJ.append(float(df["total_energy_kJ"].iloc[-1]))
            total_load.append(float(df["total_job_unit"].iloc[-1]))
    total_energy_J = np.array(total_energy_kJ) * 1000  # kJ → J
    total_load = np.array(total_load)
    totals = np.divide(total_energy_J, total_load)

    positions = np.arange(len(names))
    plt.figure(figsize=(6, 4))

    plt.plot(positions, totals, marker="o", linestyle="-", color="tab:blue", linewidth=2)

    for pos, total in zip(positions, totals):
        plt.text(pos, total, f"{total:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(positions, names, rotation=20)
    plt.ylabel("Energy by Load (J/size)")
    plt.title("Final Energy by Load per Run")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def average_latency_by_config(job_dict: Dict[str, pd.DataFrame], outpath: str):
    names, avg_service_time, avg_t_nf = [], [], []
    for name, df in job_dict.items():
        if "latency_s" not in df.columns:
            continue
        d = df.copy()
        service_time = pd.to_numeric(d["latency_s"], errors="coerce").to_numpy()
        mean_service_time = np.mean(service_time)

        if "size" in d.columns and len(d) > 0:
            job_sizes = pd.to_numeric(d["size"], errors="coerce").to_numpy()
            latency_by_size = np.divide(service_time, job_sizes)
            mean_t_nf = np.mean(latency_by_size)
        else:
            mean_t_nf = np.nan

        names.append(name)
        avg_service_time.append(mean_service_time)
        avg_t_nf.append(mean_t_nf)

    fig, ax1 = plt.subplots(figsize=(7, 5))

    positions = np.arange(len(names))
    color_service_time = "tab:blue"
    color_t_nf = "tab:orange"

    # Average Service Time
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Average Service Time (s)", color=color_service_time)
    ax1.plot(positions, avg_service_time, marker="o", color=color_service_time, label="Avg latency")
    ax1.tick_params(axis="y", labelcolor=color_service_time)
    for x, y in zip(positions, avg_service_time):
        ax1.text(x, y + 0.02 * max(avg_service_time), f"{y:.0f}", color=color_service_time, ha="center", va="bottom", fontsize=9)

    # Average T(n,f)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average T(n,f) (job_size/s)", color=color_t_nf)
    ax2.plot(positions, avg_t_nf, marker="s", linestyle="--", color=color_t_nf, label="Avg throughput")
    ax2.tick_params(axis="y", labelcolor=color_t_nf)
    for x, y in zip(positions, avg_t_nf):
        ax2.text(x, y - 0.02 * max(avg_t_nf), f"{y:.5f}", color=color_t_nf, ha="center", va="top",
                 fontsize=9)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(names, rotation=20)
    plt.title("Average Service Time and T(n,f) by Configuration")

    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_completed_jobs_by_type(
    job_dict: Dict[str, pd.DataFrame],
    outpath: str,
    kind: str = "grouped"   # "grouped" or "stacked"
):
    """
    Draw number of completed jobs on algorithms.

    Parameters
    ----------
    kind : {'grouped', 'stacked'}, default='grouped'
        - 'grouped' : two bars (training, inference) side by side.
        - 'stacked' : training/inference stacked vertically.
    """
    rows = []
    for name, df in job_dict.items():
        if df is None or len(df) == 0:
            train_cnt, infer_cnt = 0, 0
        else:
            vc = df["type"].value_counts(dropna=False)
            train_cnt  = int(vc.get("training", 0))
            infer_cnt  = int(vc.get("inference", 0))
        rows.append({"config": name, "training": train_cnt, "inference": infer_cnt})

    agg = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)

    plt.figure()

    names = agg["config"].tolist()
    train = agg["training"].to_numpy()
    infer = agg["inference"].to_numpy()
    pos   = np.arange(len(names))

    if kind.lower() == "stacked":
        # Stacked bar
        p1 = plt.bar(pos, train, label="Training")
        p2 = plt.bar(pos, infer, bottom=train, label="Inference")

        for i, (trn, inf) in enumerate(zip(train, infer)):
            (trn > 0) and plt.text(i, trn/2, str(trn), ha="center", va="center", fontsize=9)  # train
            (inf > 0) and plt.text(i, trn + inf/2, str(inf), ha="center", va="center", fontsize=9) # inference
            total = trn + inf
            plt.text(i, total, f"{total}", ha="center", va="bottom", fontsize=9)
    else:
        # Grouped bar (default)
        width = 0.4
        p1 = plt.bar(pos - width/2, train, width, label="Training")
        p2 = plt.bar(pos + width/2, infer, width, label="Inference")

        for i, trn in enumerate(train):
            (trn > 0) and plt.text(pos[i] - width/2, trn, str(trn), ha="center", va="bottom", fontsize=9)
        for i, inf in enumerate(infer):
            (inf > 0) and plt.text(pos[i] + width/2, inf, str(inf), ha="center", va="bottom", fontsize=9)

    plt.xticks(pos, names, rotation=20)
    plt.ylabel("Number of completed jobs")
    plt.title("Completed jobs per config by type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot multiple simulator runs (cluster/job CSVs) with matplotlib (no seaborn).")
    ap.add_argument("--run", action="append", default=[],
                    help="Khai báo 1 run: NAME=DIR; trong DIR có cluster_log.csv & job_log.csv. "
                         "Ví dụ: baseline=./runs/baseline (có thể dùng nhiều --run)")
    ap.add_argument("--outdir", type=str, default="./figs", help="Thư mục output để lưu các hình.")
    ap.add_argument("--bin", type=float, default=5.0, help="Kích thước bin (giây) cho throughput.")
    ap.add_argument("--scaledown", type=int, default=1, help="Bước nhảy khi đọc hàng trong log. Dùng khi muốn downsample.")
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("Cần ít nhất một --run NAME=DIR")
    os.makedirs(args.outdir, exist_ok=True)

    # Load
    cluster_by_run: Dict[str, pd.DataFrame] = {}
    jobs_by_run: Dict[str, pd.DataFrame] = {}
    agg_by_run: Dict[str, pd.DataFrame] = {}

    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Run '{spec}' không hợp lệ. Dùng NAME=DIR.")
        name, d = spec.split("=", 1)
        cl, jb = load_run(d, scaledown=args.scaledown)
        cluster_by_run[name] = cl
        jobs_by_run[name] = jb
        agg_by_run[name] = aggregate_cluster(cl)

    total_infer = sum(
        len(df[df["type"] == "inference"]) if "type" in df.columns else 0
        for df in jobs_by_run.values()
    )
    has_infer = total_infer > 0
    # 1) total power over time
    plot_lines_over_time(
        {k: v for k, v in agg_by_run.items()},
        x="time_s", y="total_power_W",
        ylabel="Total power (W)",
        title="Total power vs time",
        outpath=os.path.join(args.outdir, "total_power_vs_time.png")
    )

    # 2) cumulative energy over time
    plot_lines_over_time(
        {k: v for k, v in agg_by_run.items()},
        x="time_s", y="total_energy_kJ",
        ylabel="Total energy (kJ)",
        title="Cumulative energy vs time",
        outpath=os.path.join(args.outdir, "cumulative_energy_vs_time.png")
    )

    # 3) utilization over time
    plot_lines_over_time(
        {k: v for k, v in agg_by_run.items()},
        x="time_s", y="util",
        ylabel="Overall GPU utilization",
        title="Utilization vs time",
        outpath=os.path.join(args.outdir, "utilization_vs_time.png")
    )

    # 4) queues over time
    plot_queues_over_time(
        {k: v for k, v in agg_by_run.items()},
        outpath=os.path.join(args.outdir, "queue_lengths_vs_time.png"),
        has_infer=has_infer
    )

    # 5) latency histograms
    if has_infer:
        plot_latency_hist(jobs_by_run, job_type="inference",
                          outpath=os.path.join(args.outdir, "latency_hist_inference.png"))
    plot_latency_hist(jobs_by_run, job_type="training",
                      outpath=os.path.join(args.outdir, "latency_hist_training.png"))

    # 6) energy vs latency scatter
    plot_energy_vs_latency(jobs_by_run, outpath=os.path.join(args.outdir, "energy_per_job_scatter.png"))

    # 7) total energy bar
    plot_total_energy_bar(agg_by_run, outpath=os.path.join(args.outdir, "total_energy_bar.png"))

    # 8) throughput vs time
    plot_throughput(jobs_by_run, outpath=os.path.join(args.outdir, "throughput_vs_time.png"),
                    bin_size_s=float(args.bin))

    # 9) energy by load
    plot_energy_by_load(agg_by_run, outpath=os.path.join(args.outdir, "energy_by_load.png"))

    # 10) average latency & throughput of each config
    average_latency_by_config(jobs_by_run, outpath=os.path.join(args.outdir, "avg_latency_throughput.png"))

    # 11) number of jobs completed
    plot_completed_jobs_by_type(jobs_by_run, outpath=os.path.join(args.outdir, "completed_jobs_by_type.png"),
                                kind="grouped")

    print(f"Saved figures to: {args.outdir}")


if __name__ == "__main__":
    main()
