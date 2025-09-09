from .coeffs import TrainPowerCoeffs
from .models import DataCenter


def gpu_power_w(f_ghz: float, coeffs: TrainPowerCoeffs) -> float:
    f = max(0.0, float(f_ghz))
    return coeffs.alpha_p * (f ** 3) + coeffs.beta_p * f + coeffs.gamma_p


def task_power_w(n_gpus: int, f: float, coeffs: TrainPowerCoeffs) -> float:
    """Công suất job (W) = n * P_gpu(f). n ép về số nguyên không âm."""
    n = max(0, int(n_gpus))
    return n * gpu_power_w(f, coeffs)
