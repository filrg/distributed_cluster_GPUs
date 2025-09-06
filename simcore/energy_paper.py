from .coeffs import TrainPowerCoeffs

def gpu_power_w(f_ghz: float, coeffs: TrainPowerCoeffs) -> float:
    f = max(0.0, float(f_ghz))
    return coeffs.alpha_p * (f ** 3) + coeffs.beta_p * f + coeffs.gamma_p

def task_power_w(n_gpus: int, f_ghz: float, coeffs: TrainPowerCoeffs) -> float:
    return max(0, int(n_gpus)) * gpu_power_w(f_ghz, coeffs)
