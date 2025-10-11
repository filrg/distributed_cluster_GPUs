import random
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
    w_energy: float = 0.0
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
    names = list(dcs.keys())

    seleted_name = random.choice(names)
    Lnet, path, _, _ = graph.shortest_path_latency(ing.name, seleted_name)
    best = (seleted_name, Lnet, path, 0.0)
    #print(f"Selected DC = {best} - {ing.name}")
    return best
