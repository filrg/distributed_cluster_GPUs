import argparse, os
from simcore.simulator_paper_multi import MultiIngressPaperSimulator
from configs.paper_config import (
    build_dcs, build_arrivals, build_policy, build_paper_coeffs,
    build_ingresses_and_topology, build_carbon_intensity, build_router_policy, build_energy_price
)
from simcore.validators import validate_gpus
from simcore.logger_config import get_logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Geo GPU Simulator (paper-style, multi-ingress)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core ---
    p.add_argument(
        "--duration", type=float, default=180.0,
        help="Tổng thời gian mô phỏng (giây). Ví dụ 600 = chạy 10 phút."
    )
    p.add_argument(
        "--policy", type=str, default="energy_aware",
        choices=["energy_aware", "perf_first"],
        help="Chính sách cấp phát trong từng DC: energy_aware = thiên về tiết kiệm năng lượng; "
             "perf_first = ưu tiên hiệu năng/độ trễ."
    )
    p.add_argument(
        "--log-interval", type=float, default=5.0,
        help="Chu kỳ ghi log cluster/job (giây). Nên ≥ 1.0 để log gọn."
    )
    p.add_argument(
        "--log-path", type=str, default=None,
        help="Log path"
    )
    p.add_argument(
        "--seed", type=int, default=123,
        help="Random seed."
    )
    p.add_argument(
        "--progress", default=True,
        help="Hiển thị thanh tiến trình (tqdm) theo thời gian mô phỏng. Tắt bằng --no-progress nếu muốn chạy yên lặng."
    )

    # --- Arrivals (inference) ---
    p.add_argument(
        "--inf-mode", type=str, default="sinusoid",
        choices=["poisson", "sinusoid", "off"],
        help="Quy luật đến của yêu cầu inference: poisson = khoảng cách đến ~ Exp(λ); "
             "sinusoid = λ(t)=rate*[1+amp*sin(2π t/period)], cắt về 0 nếu âm."
    )
    p.add_argument(
        "--inf-rate", type=float, default=6.0,
        help="Tốc độ trung bình của luồng inference (yêu cầu/giây). Lưu ý: với sinusoid, giá trị trung bình vẫn = rate."
    )
    p.add_argument(
        "--inf-amp", type=float, default=0.6,
        help="Biên độ tương đối cho sinusoid (không đơn vị). "
             "0 = phẳng; 0.6 = dao động ±60%% quanh rate; >1 có pha bị cắt về 0."
    )
    p.add_argument(
        "--inf-period", type=float, default=300.0,
        help="Chu kỳ của sinusoid cho inference (giây). Ví dụ 300 = 5 phút."
    )

    # --- Arrivals (training) ---
    p.add_argument(
        "--trn-mode", type=str, default="poisson",
        choices=["poisson", "sinusoid", "off"],
        help="Quy luật đến của yêu cầu training (mô tả như inference)."
    )
    p.add_argument(
        "--trn-rate", type=float, default=0.3,
        help="Tốc độ trung bình của luồng training (yêu cầu/giây). Thường nhỏ vì mỗi job training dài."
    )

    # Thuật toán / controller
    p.add_argument("--algo", type=str, default="baseline",
                   choices=[
                       "baseline", "cap_uniform", "cap_greedy",
                       "joint_nf", "bandit", "carbon_cost",
                       "eco_route", "rl_energy", "rl_energy_adv", "rl_energy_upgr",
                       "debug"
                   ])
    p.add_argument(
        "--elastic-scaling", type=bool, default=False,
        help="Enable elastic scaling, hiện chỉ dùng cho RL."
    )
    p.add_argument(
        "--power-cap", type=float, default=0.0,
        help="Ngưỡng công suất tổng toàn hệ thống (Watt). Chỉ dùng với cap_uniform/cap_greedy; ≤0 = tắt controller."
    )
    p.add_argument(
        "--control-interval", type=float, default=5.0,
        help="Chu kỳ kích hoạt controller (giây). "
             "Dùng cho cap_* và có thể dùng cho carbon_cost nếu muốn downclock theo giờ/giá."
    )
    p.add_argument("--eco-objective", type=str, default="energy",
                   choices=["energy", "carbon", "cost"],
                   help="Mục tiêu cho eco_route: energy = min năng lượng; carbon = min E*CI; cost = min E*kWh*price.")
    # Hyperparams RL
    p.add_argument("--rl-mode", type=str, default="weighted",
                   choices=["weighted", "constrained"], help="Chế độ tính reward cho RL.")
    p.add_argument("--rl-alpha", type=float, default=0.1, help="Tốc độ học Q-learning (α).")
    p.add_argument("--rl-gamma", type=float, default=0.0, help="Hệ số chiết khấu (γ). 0.0 = contextual (1-step).")
    p.add_argument("--rl-eps", type=float, default=0.2, help="Xác suất khám phá ε (epsilon-greedy).")
    p.add_argument("--rl-eps-decay", type=float, default=0.995, help="Hệ số giảm ε mỗi lần cập nhật.")
    p.add_argument("--rl-eps-min", type=float, default=0.02, help="Ngưỡng dưới của ε.")
    p.add_argument("--rl-n-cand", type=int, default=2,
                   help="Số mức n (GPU per job) ứng viên để agent chọn (1..n_cand).")
    # upgraded RL
    p.add_argument("--rl-tau", type=float, default=0.1, help="Nhiệt độ softmax (nhỏ → khai thác nhiều).")
    p.add_argument("--rl-clip-grad", type=float, default=5.0, help="Ngưỡng clip gradient theo norm.")
    p.add_argument("--rl-baseline-beta", type=float, default=0.01, help="Hệ số cập nhật baseline reward.")

    # debug params
    p.add_argument("--num_fixed_gpus", type=int, default=1, help="Số GPUs cố định cho 1 job.")
    p.add_argument("--fixed_freq", type = float, default=None, help="Tần số GPU cố định cho 1 job.")

    # === Extra knobs for the upgraded RL mode (safe defaults) ===
    p.add_argument('--upgr-buffer', type=int, default=200_000, help='Replay capacity for upgraded RL')
    p.add_argument('--upgr-batch', type=int, default=256, help='Batch size for upgraded RL')
    p.add_argument('--upgr-warmup', type=int, default=1_000, help='Warmup transitions before learning')
    p.add_argument('--upgr-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Constraints (optional). Use your own units from the simulator.
    p.add_argument('--sla_p99_ms', type=float, default=500.0, help='Target p99 latency (ms) as a constraint')
    p.add_argument('--energy_budget_j', type=float, default=None, help='Cumulative energy budget (J) as a constraint')

    return p.parse_args()


def main():
    args = parse_args()
    dcs = build_dcs()
    #dcs = build_dc()
    warnings = validate_gpus((dc.gpu_type for dc in dcs.values()), strict=False)
    for m in warnings:
        print("[GPU VALIDATION]", m)

    ingresses, graph = build_ingresses_and_topology()
    #ingresses, graph = build_ingress_and_topology()
    arrival_inf, arrival_trn = build_arrivals(inf_mode=args.inf_mode, inf_rate=args.inf_rate,
                                              inf_amp=args.inf_amp, inf_period=args.inf_period,
                                              trn_mode=args.trn_mode, trn_rate=args.trn_rate)
    policy = build_policy(name=args.policy)
    coeffs = build_paper_coeffs(dcs)
    carbon = build_carbon_intensity()
    price = build_energy_price()
    router = build_router_policy()
    elastic_scaling = True if args.elastic_scaling == "True" else False
    if args.log_path:
        norm_path = os.path.normpath(args.log_path)
        out_dir = os.path.join(norm_path, args.algo) if os.sep not in norm_path else norm_path
    else:
        out_dir = os.getcwd()
    logger = get_logger(log_dir=out_dir)

    sim = MultiIngressPaperSimulator(
        ingresses=ingresses, dcs=dcs, graph=graph,
        arrival_inf=arrival_inf, arrival_train=arrival_trn,
        router_policy=router, coeffs_map=coeffs, carbon_intensity=carbon, energy_price=price,
        policy=policy, sim_duration=args.duration,
        log_interval=args.log_interval, log_path=out_dir,
        rng_seed=args.seed,
        algo=args.algo, elastic_scaling=elastic_scaling,
        power_cap=args.power_cap, control_interval=args.control_interval,
        show_progress=args.progress,
        # RL params
        rl_mode=args.rl_mode,
        rl_alpha=args.rl_alpha, rl_gamma=args.rl_gamma,
        rl_eps=args.rl_eps, rl_eps_decay=args.rl_eps_decay, rl_eps_min=args.rl_eps_min,
        rl_n_cand=args.rl_n_cand,
        # improved RL algo
        rl_tau=args.rl_tau, rl_clip_grad=args.rl_clip_grad, rl_baseline_beta=args.rl_baseline_beta,
        # upgraded RL alog
        energy_budget_j=args.energy_budget_j, sla_p99_ms=args.sla_p99_ms,
        upgr_batch=args.upgr_batch, upgr_warmup=args.upgr_warmup, upgr_buffer=args.upgr_buffer,
        # debug
        num_fixed_gpus=args.num_fixed_gpus, fixed_freq=args.fixed_freq, logger=logger
    )
    sim.run()
    print(f"Done. ({args.algo}) Logs: cluster_log.csv, job_log.csv")


if __name__ == "__main__":
    main()
