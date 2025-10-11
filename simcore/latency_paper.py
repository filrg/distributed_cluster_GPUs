from .coeffs import TrainLatencyCoeffs


def step_time_s(n_gpus: int, f_ghz: float, coeffs: TrainLatencyCoeffs) -> float:
    n = max(1, int(n_gpus))
    f = max(1e-9, float(f_ghz))
    return coeffs.alpha_t + coeffs.beta_t / (f * n) + coeffs.gamma_t * n
