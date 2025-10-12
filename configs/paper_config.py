from typing import Dict
from simcore.models import GPUType, DataCenter
from simcore.arrivals import ArrivalConfig
from simcore.policy import PolicyConfig
from simcore.coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from simcore.network import Graph, Ingress
from simcore.router import RouterPolicy


def build_dc():
    H100_PCIe = GPUType("H100-PCIe", p_idle=45.0, p_peak=350.0, p_sleep=28.0, alpha=3.0)

    return {
        "us-west": DataCenter("us-west", gpu_type=H100_PCIe, total_gpus=128,
                              freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                              default_freq=1.0, power_gating=True)
    }


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
        "us-west": DataCenter("us-west", gpu_type=H100_PCIe, total_gpus=120,
                              freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                              default_freq=1.0, power_gating=True),
        "us-east": DataCenter("us-east", gpu_type=A100_PCIe, total_gpus=120,
                              freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                              default_freq=1.0, power_gating=True),
        "eu-west": DataCenter("eu-west", gpu_type=L40S, total_gpus=96,
                              freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                              default_freq=1.0, power_gating=True),
        "eu-central": DataCenter("eu-central", gpu_type=H100_SXM, total_gpus=144,
                                 freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                 default_freq=1.0, power_gating=True),
        "ap-southeast": DataCenter("ap-southeast", gpu_type=L4, total_gpus=120,
                                   freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                   default_freq=1.0, power_gating=True),
        "ap-northeast": DataCenter("ap-northeast", gpu_type=H200_PCIe, total_gpus=144,
                                   freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                   default_freq=1.0, power_gating=True),
        "sa-east": DataCenter("sa-east", gpu_type=A30, total_gpus=96,
                              freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                              default_freq=1.0, power_gating=True),
        "me-central": DataCenter("me-central", gpu_type=A10, total_gpus=96,
                                 freq_levels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                                 default_freq=1.0, power_gating=True),
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
    """Example coefficients (approximate, scaled by GPU tier).
    Replace with real calibrated numbers per model/GPU.
    """
    coeffs = {}

    # us-west: H100-PCIe (high-end)
    # tweak back to first coeff, last time adjustment due to wrong latency/computing performance model
    coeffs[("us-west", "training")] = (
        TrainPowerCoeffs(75.0, 80.0, 110.0),
        TrainLatencyCoeffs(0.0045, 0.032, 0.0012) # single DC config: (0.0005, 0.05, 0.0003)
    )
    coeffs[("us-west", "inference")] = (
        TrainPowerCoeffs(95.0, 20.0, 97.0),
        TrainLatencyCoeffs(0.0090, 0.0018, 0.0007) # single DC config: (0.002, 0.004, 0.0001)
    )

    # us-east: A100-PCIe (mid-high)
    coeffs[("us-east", "training")] = (
        TrainPowerCoeffs(65.0, 60.0, 90.0),
        TrainLatencyCoeffs(0.0050, 0.038, 0.0014)
    )
    coeffs[("us-east", "inference")] = (
        TrainPowerCoeffs(85.0, 18.0, 80.0),
        TrainLatencyCoeffs(0.0080, 0.0020, 0.0009)
    )

    # eu-west: L40S (mid)
    coeffs[("eu-west", "training")] = (
        TrainPowerCoeffs(55.0, 40.0, 70.0),
        TrainLatencyCoeffs(0.0060, 0.045, 0.0018)
    )
    coeffs[("eu-west", "inference")] = (
        TrainPowerCoeffs(70.0, 15.0, 60.0),
        TrainLatencyCoeffs(0.0050, 0.020, 0.0010)
    )

    # eu-central: H100-SXM (HPC class)
    coeffs[("eu-central", "training")] = (
        TrainPowerCoeffs(90.0, 85.0, 120.0),
        TrainLatencyCoeffs(0.0042, 0.030, 0.0011)
    )
    coeffs[("eu-central", "inference")] = (
        TrainPowerCoeffs(100.0, 22.0, 100.0),
        TrainLatencyCoeffs(0.0085, 0.0017, 0.0007)
    )

    # ap-southeast: L4 (low power inference-focused)
    coeffs[("ap-southeast", "training")] = (
        TrainPowerCoeffs(45.0, 20.0, 40.0),
        TrainLatencyCoeffs(0.0065, 0.060, 0.0022)
    )
    coeffs[("ap-southeast", "inference")] = (
        TrainPowerCoeffs(40.0, 12.0, 35.0),
        TrainLatencyCoeffs(0.0045, 0.025, 0.0012)
    )

    # ap-northeast: H200-PCIe (next-gen high-end)
    coeffs[("ap-northeast", "training")] = (
        TrainPowerCoeffs(95.0, 90.0, 125.0),
        TrainLatencyCoeffs(0.0040, 0.029, 0.0010)
    )
    coeffs[("ap-northeast", "inference")] = (
        TrainPowerCoeffs(105.0, 25.0, 105.0),
        TrainLatencyCoeffs(0.0080, 0.0016, 0.0006)
    )

    # sa-east: A30 (mid-tier)
    coeffs[("sa-east", "training")] = (
        TrainPowerCoeffs(50.0, 35.0, 65.0),
        TrainLatencyCoeffs(0.0062, 0.050, 0.0019)
    )
    coeffs[("sa-east", "inference")] = (
        TrainPowerCoeffs(65.0, 14.0, 55.0),
        TrainLatencyCoeffs(0.0055, 0.022, 0.0011)
    )

    # me-central: A10 (entry-tier)
    coeffs[("me-central", "training")] = (
        TrainPowerCoeffs(40.0, 25.0, 50.0),
        TrainLatencyCoeffs(0.0068, 0.055, 0.0023)
    )
    coeffs[("me-central", "inference")] = (
        TrainPowerCoeffs(55.0, 12.0, 45.0),
        TrainLatencyCoeffs(0.0050, 0.023, 0.0012)
    )

    return coeffs


def build_ingress_and_topology():
    ingress = {
        "gw-us-west": Ingress("gw-us-west", region="US")
    }

    g = Graph()
    g.add_edge("gw-us-west", "us-west", 12)
    g.add_edge("us-west", "gw-us-west", 12)

    return ingress, g

def build_ingresses_and_topology():
    # Ingress nodes
    ingresses = {
        "gw-us-west": Ingress("gw-us-west", region="US"),
        "gw-us-east": Ingress("gw-us-east", region="US"),
        "gw-eu-west": Ingress("gw-eu-west", region="EU"),
        "gw-eu-central": Ingress("gw-eu-central", region="EU"),
        "gw-ap-southeast": Ingress("gw-ap-southeast", region="APAC"),
        "gw-ap-northeast": Ingress("gw-ap-northeast", region="APAC"),
        "gw-sa-east": Ingress("gw-sa-east", region="SA"),
        "gw-me-central": Ingress("gw-me-central", region="ME"),
    }

    # WAN graph
    g = Graph()

    # Kết nối ingress <-> DC (latency minh họa, ms)
    # US West
    g.add_edge("gw-us-west", "us-west", 12)
    g.add_edge("us-west", "gw-us-west", 12)
    g.add_edge("gw-us-west", "us-east", 70)
    g.add_edge("us-east", "gw-us-west", 70)
    g.add_edge("gw-us-west", "eu-central", 110)
    g.add_edge("eu-central", "gw-us-west", 110)
    g.add_edge("gw-us-west", "ap-southeast", 150)
    g.add_edge("ap-southeast", "gw-us-west", 150)

    # US East
    g.add_edge("gw-us-east", "us-east", 10)
    g.add_edge("us-east", "gw-us-east", 10)
    g.add_edge("gw-us-east", "us-west", 70)
    g.add_edge("us-west", "gw-us-east", 70)
    g.add_edge("gw-us-east", "eu-west", 90)
    g.add_edge("eu-west", "gw-us-east", 90)
    g.add_edge("gw-us-east", "sa-east", 110)
    g.add_edge("sa-east", "gw-us-east", 110)

    # EU West
    g.add_edge("gw-eu-west", "eu-west", 10)
    g.add_edge("eu-west", "gw-eu-west", 10)
    g.add_edge("gw-eu-west", "eu-central", 20)
    g.add_edge("eu-central", "gw-eu-west", 20)
    g.add_edge("gw-eu-west", "us-east", 90)
    g.add_edge("us-east", "gw-eu-west", 90)
    g.add_edge("gw-eu-west", "ap-northeast", 190)
    g.add_edge("ap-northeast", "gw-eu-west", 190)

    # EU Central
    g.add_edge("gw-eu-central", "eu-central", 10)
    g.add_edge("eu-central", "gw-eu-central", 10)
    g.add_edge("gw-eu-central", "me-central", 60)
    g.add_edge("me-central", "gw-eu-central", 60)
    g.add_edge("gw-eu-central", "ap-southeast", 170)
    g.add_edge("ap-southeast", "gw-eu-central", 170)

    # AP Southeast
    g.add_edge("gw-ap-southeast", "ap-southeast", 8)
    g.add_edge("ap-southeast", "gw-ap-southeast", 8)
    g.add_edge("gw-ap-southeast", "ap-northeast", 60)
    g.add_edge("ap-northeast", "gw-ap-southeast", 60)
    g.add_edge("gw-ap-southeast", "eu-central", 170)
    g.add_edge("eu-central", "gw-ap-southeast", 170)

    # AP Northeast
    g.add_edge("gw-ap-northeast", "ap-northeast", 8)
    g.add_edge("ap-northeast", "gw-ap-northeast", 8)
    g.add_edge("gw-ap-northeast", "us-west", 130)
    g.add_edge("us-west", "gw-ap-northeast", 130)
    g.add_edge("gw-ap-northeast", "eu-west", 190)
    g.add_edge("eu-west", "gw-ap-northeast", 190)

    # SA East
    g.add_edge("gw-sa-east", "sa-east", 12)
    g.add_edge("sa-east", "gw-sa-east", 12)
    g.add_edge("gw-sa-east", "us-east", 110)
    g.add_edge("us-east", "gw-sa-east", 110)
    g.add_edge("gw-sa-east", "eu-west", 150)
    g.add_edge("eu-west", "gw-sa-east", 150)

    # ME Central
    g.add_edge("gw-me-central", "me-central", 10)
    g.add_edge("me-central", "gw-me-central", 10)
    g.add_edge("gw-me-central", "eu-central", 60)
    g.add_edge("eu-central", "gw-me-central", 60)
    g.add_edge("gw-me-central", "ap-southeast", 120)
    g.add_edge("ap-southeast", "gw-me-central", 120)

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
