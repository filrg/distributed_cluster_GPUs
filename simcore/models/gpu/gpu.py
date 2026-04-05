from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUType:
    name: str
    p_idle: float  # W, GPU rảnh nhưng bật xung
    p_peak: float  # W, phần động tại f=1.0 (cộng vào idle)
    p_sleep: float  # W, khi power-gate (DRS)
    alpha: float = 3.0  # công suất động ~ f^alpha
    tdp: Optional[float] = None  # TDP/TBP khai báo (W); dùng cho validator