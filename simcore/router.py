from dataclasses import dataclass
from typing import Dict, Tuple
from .models import DataCenter
from .coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from .latency_paper import step_time_s
from .energy_paper import gpu_power_w
from .network import Graph, Ingress

@dataclass
class RouterPolicy:
    """Trọng số chọn DC: score = wE*E1 + wL*Lnet + wC*(E1*CI)."""
    w_energy: float = 1.0
    w_latency: float = 1.0
    w_carbon: float = 0.0   # dùng khi có carbon_intensity
    d_choices: int = 0      # power-of-d: 0 = xét hết

def select_dc(ing: Ingress,
              job_type: str,
              dcs: Dict[str, DataCenter],
              coeffs_map: Dict[Tuple[str, str], Tuple[TrainPowerCoeffs, TrainLatencyCoeffs]],
              graph: Graph,
              carbon_intensity: Dict[str, float],
              policy: RouterPolicy):
    """Chọn DC tối thiểu hóa điểm số composite. Trả về (dc_name, net_latency_s, path, score)."""
    # ứng viên
    names = list(dcs.keys())
    if policy.d_choices and policy.d_choices < len(names):
        import random
        names = random.sample(names, policy.d_choices)

    best = (None, float('inf'), [], float('inf'))
    for name in names:
        dc = dcs[name]
        if (name, job_type) not in coeffs_map:
            continue
        pC, tC = coeffs_map[(name, job_type)]
        f = dc.current_freq
        # energy per unit at n=1
        T1 = step_time_s(1, f, tC)
        P1 = gpu_power_w(f, pC)
        E1 = P1 * T1
        Lnet, path, bottleneck, cost_sum = graph.shortest_path_latency(ing.name, name)
        CI = carbon_intensity.get(name, 0.0)  # gCO2/kWh, đơn vị tương đối
        score = policy.w_energy * E1 + policy.w_latency * Lnet + policy.w_carbon * (E1 * CI)
        if score < best[3]:
            best = (name, Lnet, path, score)
    if best[0] is None:
        # fallback: DC đầu tiên
        first = next(iter(dcs.keys()))
        Lnet, path, *_ = graph.shortest_path_latency(ing.name, first)
        return first, Lnet, path, 0.0
    return best
