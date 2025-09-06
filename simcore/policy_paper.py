from typing import Iterable, Tuple
from .coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from .energy_paper import gpu_power_w
from .latency_paper import step_time_s

def best_energy_freq(n: int, freq_levels: Iterable[float],
                     p_coeffs: TrainPowerCoeffs, t_coeffs: TrainLatencyCoeffs) -> float:
    best_f, best_e = None, float('inf')
    for f in freq_levels:
        T = step_time_s(n, f, t_coeffs)
        P = n * gpu_power_w(f, p_coeffs)
        E = P * T
        if E < best_e:
            best_e, best_f = E, f
    return best_f if best_f is not None else max(freq_levels)

def keep_perf_when_expand(n0: int, f0: float, n1: int,
                          t_coeffs: TrainLatencyCoeffs,
                          freq_levels: Iterable[float]) -> float:
    T_target = step_time_s(n0, max(1e-9, f0), t_coeffs)
    denom = T_target - t_coeffs.alpha_t - t_coeffs.gamma_t * max(1, int(n1))
    if denom <= 1e-12:
        return f0
    f1_cont = t_coeffs.beta_t / denom
    levels = list(freq_levels)
    f1 = min(levels, key=lambda x: abs(x - f1_cont))
    return f1

def energy_tuple(n: int, f: float,
                 p_coeffs: TrainPowerCoeffs, t_coeffs: TrainLatencyCoeffs) -> Tuple[float, float, float]:
    T = step_time_s(n, f, t_coeffs)
    P = n * gpu_power_w(f, p_coeffs)
    E = P * T
    return (T, P, E)
