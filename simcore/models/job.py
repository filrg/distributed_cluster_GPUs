from dataclasses import dataclass
from typing import Optional

@dataclass
class Job:
    jid: int
    ingress: str
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