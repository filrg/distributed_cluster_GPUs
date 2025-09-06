from .models import DataCenter

def instantaneous_power_w(dc: DataCenter) -> float:
    f = dc.current_freq
    gt = dc.gpu_type
    active = dc.busy_gpus
    idle = dc.total_gpus - active
    # Active: p_idle + p_peak * f^alpha
    p_active = active * (gt.p_idle + gt.p_peak * (f ** gt.alpha))
    # Idle: power-gating nếu cho phép
    p_idle = idle * (gt.p_sleep if dc.power_gating else gt.p_idle)
    return p_active + p_idle
