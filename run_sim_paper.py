import argparse
from simcore.simulator_paper_multi import MultiIngressPaperSimulator
from configs.paper_config import (
    build_dcs, build_arrivals, build_policy, build_paper_coeffs,
    build_ingresses_and_topology, build_carbon_intensity, build_router_policy, build_energy_price
)
from simcore.validators import validate_gpus


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
        choices=["poisson", "sinusoid"],
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
        choices=["poisson", "sinusoid"],
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
                       "joint_nf", "bandit", "carbon_cost"
                   ])
    p.add_argument(
        "--power-cap", type=float, default=0.0,
        help="Ngưỡng công suất tổng toàn hệ thống (Watt). Chỉ dùng với cap_uniform/cap_greedy; ≤0 = tắt controller."
    )
    p.add_argument(
        "--control-interval", type=float, default=5.0,
        help="Chu kỳ kích hoạt controller (giây). "
             "Dùng cho cap_* và có thể dùng cho carbon_cost nếu muốn downclock theo giờ/giá."
    )

    return p.parse_args()


def main():
    args = parse_args()
    dcs = build_dcs()

    warnings = validate_gpus((dc.gpu_type for dc in dcs.values()), strict=False)
    for m in warnings:
        print("[GPU VALIDATION]", m)

    ingresses, graph = build_ingresses_and_topology()
    arrival_inf, arrival_trn = build_arrivals(inf_mode=args.inf_mode, inf_rate=args.inf_rate,
                                              inf_amp=args.inf_amp, inf_period=args.inf_period,
                                              trn_mode=args.trn_mode, trn_rate=args.trn_rate)
    policy = build_policy(name=args.policy)
    coeffs = build_paper_coeffs(dcs)
    carbon = build_carbon_intensity()
    price = build_energy_price()
    router = build_router_policy()

    sim = MultiIngressPaperSimulator(
        ingresses=ingresses, dcs=dcs, graph=graph,
        arrival_inf=arrival_inf, arrival_train=arrival_trn,
        router_policy=router, coeffs_map=coeffs, carbon_intensity=carbon, energy_price=price,
        policy=policy, sim_duration=args.duration,
        log_interval=args.log_interval, rng_seed=args.seed,
        algo=args.algo, power_cap=args.power_cap, control_interval=args.control_interval,
        show_progress=args.progress
    )
    sim.run()
    print(f"Done. ({args.algo}) Logs: cluster_log.csv, job_log.csv")


if __name__ == "__main__":
    main()
