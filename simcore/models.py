from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    ev_gen: int = 0  # thế hệ event; dùng để vô hiệu hoá job_finish cũ


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

    def __post_init__(self):
        assert self.default_freq in self.freq_levels, "default_freq phải thuộc freq_levels"
        self.current_freq = self.default_freq

    @property
    def free_gpus(self) -> int:
        return self.total_gpus - self.busy_gpus

    def accrue_energy(self, now: float):
        from .energy_paper import instantaneous_power_w
        dt = max(0.0, now - self.last_energy_time)
        if dt > 0.0:
            self.energy_joules += instantaneous_power_w(self) * dt
            self.last_energy_time = now
