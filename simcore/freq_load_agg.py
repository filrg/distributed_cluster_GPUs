from dataclasses import dataclass
from typing import Iterable, List
from .coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from .latency_paper import step_time_s
from .energy_paper import gpu_power_w

@dataclass
class TaskState:
    job_id: int
    dc_name: str
    n: int
    f: float
    freq_levels: List[float]
    p_coeffs: TrainPowerCoeffs
    t_coeffs: TrainLatencyCoeffs

@dataclass
class Atom:
    # Một bước DVFS rời rạc cho một task
    rho: float         # ΔP/ΔV
    dV: float
    dP: float
    job_id: int
    dc_name: str
    f_from: float
    f_to: float

def _V(n,f,t):
    T = step_time_s(n,f,t);  return 0.0 if T<=0 else 1.0/T

def _P(n,f,p):
    return n * gpu_power_w(f,p)

def _nearest_idx(levels, f):
    return min(range(len(levels)), key=lambda i: abs(levels[i]-f))

def atoms_for_task(t: TaskState):
    lv = sorted(t.freq_levels)
    i0 = _nearest_idx(lv, t.f)
    Vi, Pi = _V(t.n, lv[i0], t.t_coeffs), _P(t.n, lv[i0], t.p_coeffs)
    up, down = [], []

    # UP: i0 -> i0+1 -> ...
    curV, curP = Vi, Pi
    for k in range(i0, len(lv)-1):
        f_from, f_to = lv[k], lv[k+1]
        V2, P2 = _V(t.n, f_to, t.t_coeffs), _P(t.n, f_to, t.p_coeffs)
        dV, dP = max(0.0, V2-curV), max(0.0, P2-curP)
        if dV>0 and dP>=0:
            up.append(Atom(rho=dP/dV, dV=dV, dP=dP, job_id=t.job_id, dc_name=t.dc_name, f_from=f_from, f_to=f_to))
        curV, curP = V2, P2

    # DOWN: i0 -> i0-1 -> ...
    curV, curP = Vi, Pi
    for k in range(i0, 0, -1):
        f_from, f_to = lv[k], lv[k-1]
        V2, P2 = _V(t.n, f_to, t.t_coeffs), _P(t.n, f_to, t.p_coeffs)
        dV, dP = max(0.0, curV - V2), max(0.0, curP - P2)  # mất V, tiết kiệm P
        if dV>0 and dP>=0:
            down.append(Atom(rho=dP/dV, dV=dV, dP=dP, job_id=t.job_id, dc_name=t.dc_name, f_from=f_from, f_to=f_to))
        curV, curP = V2, P2
    return up, down

def aggregate_with_atoms(tasks: Iterable[TaskState]):
    up_all, down_all = [], []
    for t in tasks:
        u, d = atoms_for_task(t)
        up_all.extend(u); down_all.extend(d)
    up_all.sort(key=lambda a: a.rho)     # rẻ điện nhất cho mỗi +V
    down_all.sort(key=lambda a: a.rho)   # rẻ nhất để tiết kiệm P
    return up_all, down_all
