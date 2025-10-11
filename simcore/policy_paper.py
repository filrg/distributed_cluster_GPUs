from typing import Iterable, Tuple
from .coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from .energy_paper import gpu_power_w, task_power_w
from .latency_paper import step_time_s


def best_energy_freq(n: int, freq_levels: Iterable[float],
                     p_coeffs: TrainPowerCoeffs, t_coeffs: TrainLatencyCoeffs) -> float:
    best_f, best_e = None, float('inf')
    for f in freq_levels:
        T = step_time_s(n, f, t_coeffs)
        P = task_power_w(n, f, p_coeffs)
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
    P = task_power_w(n, f, p_coeffs)
    E = P * T
    return (T, P, E)


def best_nf_grid(n_max: int, freq_levels,
                 p_coeffs: TrainPowerCoeffs, t_coeffs: TrainLatencyCoeffs,
                 objective: str = "energy",  # "energy" | "carbon"
                 carbon_intensity: float = 0.0,  # gCO2/kWh (tương đối cũng được)
                 price_kwh: float = 0.0, deadline_s=None):
    """
    Trả về (n*, f*, T*, P*, E*). f_levels là list float.
    """
    best = None
    for n in range(1, max(1, int(n_max)) + 1):
        for f in freq_levels:
            T = step_time_s(n, f, t_coeffs)  # s per unit
            P = task_power_w(n, f, p_coeffs)  # W
            E = P * T  # J per unit
            if deadline_s is not None and T > deadline_s:
                continue

            if objective == "energy":
                score = E
            elif objective == "carbon":
                score = E * carbon_intensity  # J * gCO2/kWh (relative OK)
            elif objective == "cost":
                score = (E / 3.6e6) * float(price_kwh)  # J -> kWh, rồi * giá
            else: # default = "energy"
                score = E

            cand = (score, n, f, T, P, E)
            if (best is None) or (cand[0] < best[0]):
                best = cand
    if best is None:
        # fallback: dùng n=1, f=max
        fmax = max(freq_levels)
        T = step_time_s(1, fmax, t_coeffs)
        P = gpu_power_w(fmax, p_coeffs)
        E = P * T
        return 1, fmax, T, P, E
    _, n, f, T, P, E = best
    return n, f, T, P, E
