from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


@dataclass
class Job:
    jid: int
    jtype: str  # 'inference' | 'training'
    size: float  # compute units trừu tượng
    arrival_time: float
    deadline: Optional[float] = None
    dc_name: Optional[str] = None
    gpus_assigned: int = 0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    net_latency_s: float = 0.0
    f_used: float = 0.0  # tần số hiện hành của job (nếu per-job DVFS)
    units_total: float = 0.0  # tổng đơn vị công việc (size)
    units_done: float = 0.0  # đã xử lý bao nhiêu đơn vị
    last_update: float = 0.0  # thời điểm cuối cùng ta “tiến độ hoá”
    ev_gen: int = 0  # thế hệ event; dùng để vô hiệu hoá job_finish cũ - "lazy invalidation"

    preemptible: bool = False
    preempt_count: int = 0
    total_preempt_time: float = 0
    last_checkpoint: float = 0.0


@dataclass
class PreemptedJob:
    job: Job
    preempt_time: float
    reason: str
    preempt_ckpt: dict # units_done, f_used, gpus_assigned


@dataclass
class GPUType:
    name: str
    p_idle: float  # W, GPU rảnh nhưng bật xung
    p_peak: float  # W, phần động tại f=1.0 (cộng vào idle)
    p_sleep: float  # W, khi power-gate (DRS)
    alpha: float = 3.0  # công suất động ~ f^alpha
    tdp: Optional[float] = None  # TDP/TBP khai báo (W); dùng cho validator


@dataclass
class DataCenter:
    name: str
    gpu_type: GPUType
    total_gpus: int
    freq_levels: List[float]  # ví dụ [0.6, 0.8, 1.0]
    default_freq: float = 1.0
    power_gating: bool = True

    # runtime
    current_freq: float = field(init=False)
    busy_gpus: int = field(default=0, init=False)
    running_jobs: Dict[int, Tuple[Job, int]] = field(default_factory=dict, init=False)  # jid -> (Job, gpus)
    q_inf: List[Job] = field(default_factory=list, init=False)
    q_train: List[Job] = field(default_factory=list, init=False)
    energy_joules: float = field(default=0.0, init=False)
    last_energy_time: float = field(default=0.0, init=False)
    util_gpu_time: float = 0.0  # ∑ (busy_gpus * dt)  [GPU·s]
    util_last_ts: float = 0.0  # mốc thời gian lần cuối cập nhật
    util_begin_ts: float = 0.0  # mốc bắt đầu tính trung bình

    # preempt
    preempted_jobs: List[PreemptedJob] = field(default_factory=list, init=False)
    preempt_policy: str = "fifo" # "all"

    def __post_init__(self):
        assert self.default_freq in self.freq_levels, "default_freq phải thuộc freq_levels"
        self.current_freq = self.default_freq

    @property
    def free_gpus(self) -> int:
        return self.total_gpus - self.busy_gpus

    def instantaneous_power_w(self) -> float:
        f = self.current_freq
        gt = self.gpu_type
        active = self.busy_gpus
        idle = self.total_gpus - active
        # Active: p_idle + p_peak * f^alpha
        p_active = active * (gt.p_idle + gt.p_peak * (f ** gt.alpha))
        # Idle: power-gating nếu cho phép
        p_idle = idle * (gt.p_sleep if self.power_gating else gt.p_idle)
        return p_active + p_idle

    def accrue_energy(self, now: float,
                      power_fn: Optional[Callable[['DataCenter'], float]] = None) -> None:
        """
        Tích lũy năng lượng: E += P(now) * dt.
        - power_fn(dcenter) nếu có: dùng để tính công suất (paper-style, per-job f, v.v.)
        - nếu không: fallback sang self.instantaneous_power_w() (baseline idle/sleep).
        """
        if getattr(self, "last_energy_time", 0.0) == 0.0:
            self.last_energy_time = now
            return
        dt = max(0.0, now - self.last_energy_time)
        p = power_fn(self) if power_fn else self.instantaneous_power_w()
        self.energy_joules += p * dt
        self.last_energy_time = now
