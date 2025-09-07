import argparse
from simcore.simulator_paper_multi import MultiIngressPaperSimulator
from configs.paper_config import (
    build_dcs, build_arrivals, build_policy, build_paper_coeffs,
    build_ingresses_and_topology, build_carbon_intensity, build_router_policy, build_energy_price
)
from simcore.validators import validate_gpus


def parse_args():
    p = argparse.ArgumentParser(description="Geo GPU Simulator (paper-style, multi-ingress)")
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--policy", type=str, default="energy_aware", choices=["energy_aware", "perf_first"])
    p.add_argument("--log-interval", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=123)

    # Arrivals
    p.add_argument("--inf-mode", type=str, default="sinusoid", choices=["poisson", "sinusoid"])
    p.add_argument("--inf-rate", type=float, default=6.0)
    p.add_argument("--inf-amp", type=float, default=0.6)
    p.add_argument("--inf-period", type=float, default=300.0)
    p.add_argument("--trn-mode", type=str, default="poisson", choices=["poisson", "sinusoid"])
    p.add_argument("--trn-rate", type=float, default=0.3)

    # Thuật toán / controller
    p.add_argument("--algo", type=str, default="baseline",
                   choices=[
                       "baseline", "cap_uniform", "cap_greedy",
                       "joint_nf", "bandit", "carbon_cost"
                   ])
    p.add_argument("--power-cap", type=float, default=0.0,
                   help="Watt. <=0 nghĩa là không bật power cap controller.")
    p.add_argument("--control-interval", type=float, default=5.0,
                   help="Giây. Chu kỳ kích hoạt controller (trùng với logging là an toàn).")

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
        algo=args.algo, power_cap=args.power_cap, control_interval=args.control_interval
    )
    sim.run()
    print(f"Done. ({args.algo}) Logs: cluster_log.csv, job_log.csv")


if __name__ == "__main__":
    main()
