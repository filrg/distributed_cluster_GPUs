import os
import argparse
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_run(dir_path: str, scaledown: int = 1, readafter: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def plot_lines_over_time(series_dict: Dict[str, pd.DataFrame], x, y, ylabel, title, outpath: str, show: bool = False):
    """Plot power lines over time"""
    plt.figure()
    
    is_power = "power" in y.lower()
    for name, df in series_dict.items():
        if x in df.columns and y in df.columns:
            y_data = df[y] / 1000 if is_power else df[y]
            plt.plot(df[x], y_data, label=name)
    plt.xlabel("Execution Time (s)")
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close()


def plot_queues_over_time(series_dict: Dict[str, pd.DataFrame], outpath: str, has_infer: bool = True, step: int = 86400):
    """Plot queues along with DataFrame by time step."""
    plt.figure()
    sampled_table = {}

    for name, df in series_dict.items():
        if {"time_s", "q_inf_sum", "q_train_sum"}.issubset(df.columns):
            if has_infer:
                plt.plot(df["time_s"], df["q_inf_sum"], label=f"{name}-q_inf")
            plt.plot(df["time_s"], df["q_train_sum"], label=f"{name}")

            time_points = np.arange(0, df["time_s"].max() + step, step)
            interp_train = np.interp(time_points, df["time_s"], df["q_train_sum"])
            if has_infer:
                interp_infer = np.interp(time_points, df["time_s"], df["q_inf_sum"])
                sampled_table[f"{name}_q_inf"] = interp_infer
            sampled_table[f"{name}_q_train"] = interp_train

    plt.xlabel("Execution Time (s)")
    plt.ylabel("Queue length (requests)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=160)
    plt.close()

    df_out = pd.DataFrame({"time_s": np.arange(0, df["time_s"].max() + step, step)})
    for k, v in sampled_table.items():
        df_out[k] = v

    csv_path = os.path.splitext(outpath)[0] + "_table.csv"
    df_out.to_csv(csv_path, index=False)

    return df_out


def plot_latency_histogram(job_dict: Dict[str, pd.DataFrame], job_type: str, outpath: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, df in job_dict.items():
        if {"type", "latency_s"}.issubset(df.columns):
            d = df[df["type"] == job_type]
            if len(d) > 0:
                ax.hist(d["latency_s"], bins=bins, alpha=0.5, label=f"{name} - {len(d)} jobs")

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    # ax.set_title(f"Latency Histogram — {job_type}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_latency_violin_or_boxen(job_dict: Dict[str, pd.DataFrame], job_type: str, outpath: str, kind: str = "boxen"):
    records = []
    for name, df in job_dict.items():
        if {"type", "latency_s"}.issubset(df.columns):
            d = df[df["type"] == job_type]
            if not d.empty:
                records.extend([{"Algorithm": name, "Latency (s)": v} for v in d["latency_s"]])
    all_data = pd.DataFrame(records)
    if all_data.empty:
        print(f"[WARN] No data for job_type={job_type}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    if kind == "violin":
        sns.violinplot(
            data=all_data,
            x="Algorithm",
            y="Latency (s)",
            ax=ax,
            inner="box",
            density_norm="width",  # ~ scale
            linewidth=1,
            cut=0
        )
        # Overlay mean markers
        means = all_data.groupby("Algorithm")["Latency (s)"].mean()
        ax.scatter(range(len(means)), means, color="red", marker="D", label="Mean", s=40)
        ax.legend(loc="upper right")
        # ax.set_title(f"Latency Violin — {job_type}")

    elif kind == "boxen":
        sns.boxenplot(
            data=all_data,
            x="Algorithm",
            y="Latency (s)",
            ax=ax,
            linewidth=1,
            outlier_prop=0.01  # Reduce tail noise
        )
        # ax.set_title(f"Latency Boxen — {job_type}")
    else:
        raise ValueError("kind must be 'violin' or 'boxen'")

    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)

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
    # plt.title("Energy vs Latency (predicted)")
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

    plt.xticks(positions, names, rotation=15)
    plt.ylabel("Total energy (kJ)")
    # plt.title("Final total energy per run")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_throughput(job_dict: Dict[str, pd.DataFrame], outpath: str, bin_size_s: float = 5.0, show: bool = False):
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
    # plt.title(f"Throughput vs time (bin={bin_size_s:.0f}s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    if show: plt.show()
    plt.close()


def plot_energy_by_load(agg_dict: Dict[str, pd.DataFrame], outpath: str):
    """Energy by load (J/unit)."""
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
    plt.bar(positions, totals)

    for pos, total in zip(positions, totals):
        plt.text(
            pos, total, f"{total:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.xticks(positions, names, rotation=15)
    plt.ylabel("Energy by Load (J/size)")
    # plt.title("Final Energy by Load per Run")
    plt.grid(axis="y", alpha=0.3)
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
    # color_t_nf = "tab:orange"

    # Average Service Time
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Average Service Time (s)", color=color_service_time)
    ax1.plot(positions, avg_service_time, marker="o", color=color_service_time, label="Avg latency")
    ax1.tick_params(axis="y", labelcolor=color_service_time)
    for x, y in zip(positions, avg_service_time):
        ax1.text(x, y, f"{y:.0f}", color=color_service_time, ha="center", va="bottom", fontsize=9)

    # # Average T(n,f)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Average T(n,f) (job_size/s)", color=color_t_nf)
    # ax2.plot(positions, avg_t_nf, marker="s", linestyle="--", color=color_t_nf, label="Avg throughput")
    # ax2.tick_params(axis="y", labelcolor=color_t_nf)
    # for x, y in zip(positions, avg_t_nf):
    #     ax2.text(x, y, f"{y:.5f}", color=color_t_nf, ha="center", va="top",
    #              fontsize=9)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(names, rotation=15)
    # plt.title("Average Service Time (s).")

    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_completed_jobs_by_type(job_dict: Dict[str, pd.DataFrame], outpath: str,
    kind: str = "grouped",   # "grouped" or "stacked"
):
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

    names = agg["config"].tolist()
    train = agg["training"].to_numpy()
    infer = agg["inference"].to_numpy()
    pos   = np.arange(len(names))

    has_infer = np.any(infer > 0)

    plt.figure(figsize=(7, 5))

    if has_infer:
        if kind.lower() == "stacked":
            p1 = plt.bar(pos, train, label="Training", color="tab:blue", alpha=0.8)
            p2 = plt.bar(pos, infer, bottom=train, label="Inference", color="tab:orange", alpha=0.8)

            for i, (trn, inf) in enumerate(zip(train, infer)):
                if trn > 0:
                    plt.text(i, trn / 2, str(trn), ha="center", va="center", fontsize=9)
                if inf > 0:
                    plt.text(i, trn + inf / 2, str(inf), ha="center", va="center", fontsize=9)
                total = trn + inf
                plt.text(i, total, f"{total}", ha="center", va="bottom", fontsize=9)
        else:
            width = 0.4
            p1 = plt.bar(pos - width / 2, train, width, label="Training", color="tab:blue", alpha=0.8)
            p2 = plt.bar(pos + width / 2, infer, width, label="Inference", color="tab:orange", alpha=0.8)

            for i, trn in enumerate(train):
                if trn > 0:
                    plt.text(pos[i] - width / 2, trn, str(trn), ha="center", va="bottom", fontsize=9)
            for i, inf in enumerate(infer):
                if inf > 0:
                    plt.text(pos[i] + width / 2, inf, str(inf), ha="center", va="bottom", fontsize=9)
    else:
        bars = plt.bar(pos, train, color="tab:blue", edgecolor="black", alpha=0.8, label="Jobs")
        for bar, value in zip(bars, train):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{value}", ha="center", va="bottom", fontsize=9)

    plt.xticks(pos, names, rotation=15)
    plt.ylabel("Number of completed jobs")
    # plt.title("Completed jobs per config by type")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot multiple simulator runs (cluster/job CSVs) with matplotlib.")
    ap.add_argument("--run", action="append", default=[],
                    help="Khai báo 1 run: NAME=DIR; trong DIR có cluster_log.csv & job_log.csv. "
                         "Ví dụ: baseline=./runs/baseline (có thể dùng nhiều --run)")
    ap.add_argument("--outdir", type=str, default="./figs", help="Thư mục output để lưu các hình.")
    ap.add_argument("--bin", type=float, default=5.0, help="Kích thước bin (giây) cho throughput.")
    ap.add_argument("--scaledown", type=int, default=1, help="Bước nhảy khi đọc hàng trong log. Dùng khi muốn downsample.")
    ap.add_argument("--show", action="store_true", help="Show những plot mật độ điểm lớn để điều chỉnh thủ công.")
    ap.add_argument("--pdf", action="store_true", help="Lưu ảnh ra PDF (mặc định là PNG).")
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("Cần ít nhất một --run NAME=DIR")
    os.makedirs(args.outdir, exist_ok=True)
    save_format = "pdf" if args.pdf else "png"
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
        ylabel="Total power (kW)",
        title="Total power vs time",
        outpath=os.path.join(args.outdir, f"total_power_vs_time.{save_format}"),
        show=args.show
    )

    # 2) cumulative energy over time
    plot_lines_over_time(
        {k: v for k, v in agg_by_run.items()},
        x="time_s", y="total_energy_kJ",
        ylabel="Total energy (kJ)",
        title="Cumulative energy vs time",
        outpath=os.path.join(args.outdir, f"cumulative_energy_vs_time.{save_format}"),
        show=False
    )

    # 3) utilization over time
    plot_lines_over_time(
        {k: v for k, v in agg_by_run.items()},
        x="time_s", y="util",
        ylabel="Overall GPU utilization",
        title="Utilization vs time",
        outpath=os.path.join(args.outdir, f"utilization_vs_time.{save_format}"),
        show=args.show
    )

    # 4) queues over time
    plot_queues_over_time(
        {k: v for k, v in agg_by_run.items()},
        outpath=os.path.join(args.outdir, f"queue_lengths_vs_time.{save_format}"),
        has_infer=has_infer
    )

    # 5) latency
    if has_infer:
        plot_latency_histogram(jobs_by_run, job_type="inference",
                     outpath=os.path.join(args.outdir, f"latency_hist_infer.{save_format}"))
    plot_latency_histogram(jobs_by_run, job_type="training",
                 outpath=os.path.join(args.outdir, f"latency_hist_train.{save_format}"))

    if has_infer:
        plot_latency_violin_or_boxen(jobs_by_run, job_type="inference",
                    outpath=os.path.join(args.outdir, f"latency_boxen_infer.{save_format}"))
    plot_latency_violin_or_boxen(jobs_by_run, job_type="training",
                    outpath=os.path.join(args.outdir, f"latency_boxen_train.{save_format}"))

    # 6) energy vs latency scatter
    plot_energy_vs_latency(jobs_by_run, outpath=os.path.join(args.outdir, f"energy_per_job_scatter.{save_format}"))

    # 7) total energy bar
    plot_total_energy_bar(agg_by_run, outpath=os.path.join(args.outdir, f"total_energy_bar.{save_format}"))

    # 8) throughput vs time
    plot_throughput(jobs_by_run, outpath=os.path.join(args.outdir, f"throughput_vs_time.{save_format}"),
                    bin_size_s=float(args.bin), show=args.show)

    # 9) energy by load
    plot_energy_by_load(agg_by_run, outpath=os.path.join(args.outdir, f"energy_by_load.{save_format}"))

    # 10) average latency & throughput of each config
    average_latency_by_config(jobs_by_run, outpath=os.path.join(args.outdir, f"avg_latency_throughput.{save_format}"))

    # 11) number of jobs completed
    plot_completed_jobs_by_type(jobs_by_run, outpath=os.path.join(args.outdir, f"completed_jobs_by_type.{save_format}"),
                                kind="grouped")

    print(f"Saved figures to: {args.outdir}")


if __name__ == "__main__":
    main()
