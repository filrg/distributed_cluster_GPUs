from typing import Dict
from simcore.models import GPUType, DataCenter
from simcore.arrivals import ArrivalConfig
from simcore.policy import PolicyConfig
from simcore.coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from simcore.network import Graph, Ingress
from simcore.router import RouterPolicy


def build_dcs():
    A100_SXM = GPUType("A100-SXM4", p_idle=50.0, p_peak=400.0, p_sleep=30.0, alpha=3.0)
    A100_PCIe = GPUType("A100-PCIe", p_idle=45.0, p_peak=300.0, p_sleep=28.0, alpha=3.0)

    H100_SXM = GPUType("H100-SXM5", p_idle=55.0, p_peak=700.0, p_sleep=35.0, alpha=3.0)
    H100_PCIe = GPUType("H100-PCIe", p_idle=45.0, p_peak=350.0, p_sleep=28.0, alpha=3.0)

    H200_SXM = GPUType("H200-SXM", p_idle=60.0, p_peak=700.0, p_sleep=38.0, alpha=3.0)
    H200_PCIe = GPUType("H200-PCIe", p_idle=55.0, p_peak=600.0, p_sleep=35.0, alpha=3.0)

    L4 = GPUType("L4", p_idle=15.0, p_peak=72.0, p_sleep=8.0, alpha=3.0)
    T4 = GPUType("T4", p_idle=10.0, p_peak=70.0, p_sleep=6.0, alpha=3.0)

    A10 = GPUType("A10", p_idle=20.0, p_peak=150.0, p_sleep=10.0, alpha=3.0)
    A30 = GPUType("A30", p_idle=25.0, p_peak=165.0, p_sleep=12.0, alpha=3.0)
    A40 = GPUType("A40", p_idle=40.0, p_peak=300.0, p_sleep=25.0, alpha=3.0)
    L40 = GPUType("L40", p_idle=35.0, p_peak=300.0, p_sleep=20.0, alpha=3.0)
    L40S = GPUType("L40S", p_idle=40.0, p_peak=350.0, p_sleep=25.0, alpha=3.0)

    return {
        "us-west": DataCenter("us-west", gpu_type=H100_PCIe, total_gpus=32,
                              freq_levels=[0.6, 0.8, 1.0], default_freq=1.0, power_gating=True),
        "eu-central": DataCenter("eu-central", gpu_type=A100_PCIe, total_gpus=24,
                                 freq_levels=[0.5, 0.7, 1.0], default_freq=1.0, power_gating=True),
        "ap-southeast": DataCenter("ap-southeast", gpu_type=L4, total_gpus=48,
                                   freq_levels=[0.5, 0.75, 1.0], default_freq=1.0, power_gating=True),
    }


def build_arrivals(inf_mode='sinusoid', inf_rate=6.0, inf_amp=0.6, inf_period=300.0,
                   trn_mode='poisson', trn_rate=0.3):
    arrival_inf = ArrivalConfig(mode=inf_mode, rate=inf_rate, amp=inf_amp, period=inf_period)
    arrival_trn = ArrivalConfig(mode=trn_mode, rate=trn_rate)
    return arrival_inf, arrival_trn


def build_policy(name='energy_aware', max_gpus_per_job=8, inf_priority=True,
                 dvfs_low=0.6, dvfs_high=1.0, train_scale_out_low_freq=True, reserve_inf_gpus=0):
    return PolicyConfig(name=name, max_gpus_per_job=max_gpus_per_job, inf_priority=inf_priority,
                        dvfs_low=dvfs_low, dvfs_high=dvfs_high,
                        train_scale_out_low_freq=train_scale_out_low_freq, reserve_inf_gpus=reserve_inf_gpus)


def build_paper_coeffs(dcs) -> Dict[tuple, tuple]:
    """EXAMPLE coefficients ONLY. Replace with real calibrated numbers per model/GPU."""
    coeffs = {}
    # From paper: SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving + Towards Improved Power Management in Cloud GPUs
    coeffs[("us-west", "training")] = (
        TrainPowerCoeffs(74.11, 77.71, 108.18),  # αp, βp, γp  (W)
        TrainLatencyCoeffs(0.0048, 0.0340, 0.0012)  # αt, βt, γt (s/unit)
    )

    coeffs[("us-west", "inference")] = (
        TrainPowerCoeffs(93.74, 19.47, 96.79),  # αp, βp, γp  (W)
        TrainLatencyCoeffs(0.0093, 0.00190, 0.0008)  # αt, βt, γt (s/unit)
    )

    coeffs[("eu-central", "training")] = (TrainPowerCoeffs(150.0, 30.0, 60.0), TrainLatencyCoeffs(0.005, 0.040, 0.0015))
    coeffs[("eu-central", "inference")] = (
    TrainPowerCoeffs(140.0, 28.0, 58.0), TrainLatencyCoeffs(0.003, 0.018, 0.0008))
    coeffs[("ap-southeast", "training")] = (
    TrainPowerCoeffs(80.0, 18.0, 30.0), TrainLatencyCoeffs(0.006, 0.060, 0.0020))
    coeffs[("ap-southeast", "inference")] = (
    TrainPowerCoeffs(70.0, 15.0, 25.0), TrainLatencyCoeffs(0.004, 0.025, 0.0010))
    return coeffs


def build_ingresses_and_topology():
    # Ingress nodes
    ingresses = {
        "gw-west": Ingress("gw-west", region="US"),
        "gw-eu": Ingress("gw-eu", region="EU"),
        "gw-ap": Ingress("gw-ap", region="APAC"),
    }
    # WAN graph
    g = Graph()
    # Kết nối ingress <-> DC (latency minh họa, ms)
    g.add_edge("gw-west", "us-west", 15);
    g.add_edge("us-west", "gw-west", 15)
    g.add_edge("gw-west", "eu-central", 90);
    g.add_edge("eu-central", "gw-west", 90)
    g.add_edge("gw-west", "ap-southeast", 140);
    g.add_edge("ap-southeast", "gw-west", 140)

    g.add_edge("gw-eu", "eu-central", 12);
    g.add_edge("eu-central", "gw-eu", 12)
    g.add_edge("gw-eu", "us-west", 95);
    g.add_edge("us-west", "gw-eu", 95)
    g.add_edge("gw-eu", "ap-southeast", 170);
    g.add_edge("ap-southeast", "gw-eu", 170)

    g.add_edge("gw-ap", "ap-southeast", 10);
    g.add_edge("ap-southeast", "gw-ap", 10)
    g.add_edge("gw-ap", "eu-central", 160);
    g.add_edge("eu-central", "gw-ap", 160)
    g.add_edge("gw-ap", "us-west", 130);
    g.add_edge("us-west", "gw-ap", 130)

    return ingresses, g


def build_carbon_intensity():
    # gCO2/kWh tương đối theo vùng (demo)
    return {
        "us-west": 350.0,
        "eu-central": 220.0,
        "ap-southeast": 500.0,
    }


def build_router_policy():
    # ưu tiên năng lượng + độ trễ; thêm w_carbon nếu muốn carbon-aware
    return RouterPolicy(w_energy=1.0, w_latency=0.5, w_carbon=0.0, d_choices=0)


def build_energy_price():
    # USD/kWh theo giờ (demo): giờ cao điểm đắt
    return {  # key: hour (0..23)
        **{h: 0.12 for h in range(0, 7)},
        **{h: 0.20 for h in range(7, 19)},
        **{h: 0.16 for h in range(19, 24)}
    }
