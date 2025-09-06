from dataclasses import dataclass

@dataclass
class TrainPowerCoeffs:
    """P(f) = alpha_p * f^3 + beta_p * f + gamma_p  (Watts per GPU)"""
    alpha_p: float
    beta_p: float
    gamma_p: float

@dataclass
class TrainLatencyCoeffs:
    """T(n,f) = alpha_t + beta_t / f + gamma_t * n  (seconds per unit)"""
    alpha_t: float
    beta_t: float
    gamma_t: float
