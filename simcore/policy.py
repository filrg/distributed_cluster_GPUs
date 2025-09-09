from dataclasses import dataclass
from .models import DataCenter, Job


@dataclass
class PolicyConfig:
    name: str  # 'perf_first' or 'energy_aware'
    max_gpus_per_job: int = 8
    inf_priority: bool = True
    dvfs_low: float = 0.6
    dvfs_high: float = 1.0
    train_scale_out_low_freq: bool = True
    reserve_inf_gpus: int = 0


def select_gpus_and_set_freq(dc: DataCenter, job: Job, policy: PolicyConfig) -> int:
    free = dc.free_gpus
    g = min(free, policy.max_gpus_per_job) if free > 0 else 0

    if policy.name == 'perf_first':
        if job.jtype == 'inference':
            dc.current_freq = policy.dvfs_high
            return max(1, g)
        else:
            dc.current_freq = max(dc.current_freq, policy.dvfs_high if len(dc.q_inf) > 0 else dc.default_freq)
            return max(1, g)

    elif policy.name == 'energy_aware':
        if job.jtype == 'inference':
            dc.current_freq = policy.dvfs_high
            return max(1, g)
        else:
            if policy.train_scale_out_low_freq and free >= 2:
                dc.current_freq = policy.dvfs_low
                g = min(free, policy.max_gpus_per_job)
                return max(1, g)
            else:
                dc.current_freq = max(dc.current_freq, policy.dvfs_low)
                return max(1, g)
    else:
        raise ValueError("Unknown policy name")
