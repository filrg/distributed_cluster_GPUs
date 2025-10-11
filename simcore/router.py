from dataclasses import dataclass

@dataclass
class RouterPolicy:
    """Trọng số chọn DC: score = wE*E1 + wL*Lnet + wC*(E1*CI)."""
    w_energy: float = 0.0
    w_latency: float = 1.0
    w_carbon: float = 0.0   # dùng khi có carbon_intensity
    d_choices: int = 0      # power-of-d: 0 = xét hết
