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
       - util: (sum busy) / (sum (busy+free))
       - q_inf / q_train: sums across DC
       - freq_avg: simple average of freq across DC (for reference only)
    """
    df = cl.copy()
    # Ensure column types
    for col in ["time_s", "power_W", "energy_kJ", "busy", "free", "q_inf", "q_train", "freq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_gpus"] = df["busy"] + df["free"]
    g = df.groupby("time_s", as_index=False).agg(
        total_power_W=("power_W", "sum"),
        total_energy_kJ=("energy_kJ", "sum"),
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


def plot_queues_over_time(series_dict: Dict[str, pd.DataFrame], outpath):
    # Two subplots would violate the "one chart per figure" rule in this environment.
    # So we produce a single figure with two overlaid lines per run for q_inf and q_train.
    plt.figure()
    for name, df in series_dict.items():
        if {"time_s", "q_inf_sum", "q_train_sum"}.issubset(df.columns):
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
        outpath=os.path.join(args.outdir, "queue_lengths_vs_time.png")
    )

    # 5) latency histograms
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

    print(f"Saved figures to: {args.outdir}")


if __name__ == "__main__":
    main()
