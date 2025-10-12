from typing import Dict
from dataclasses import dataclass, field
import torch


@dataclass
class PIDConfig:
    kp: float = 0.05
    ki: float = 0.01
    kd: float = 0.0
    clamp_min: float = 0.0
    clamp_max: float = 10.0


@dataclass
class ConstraintSpec:
    name: str
    target: float  # ngưỡng ràng buộc (ví dụ: p99_latency <= target)
    pid: PIDConfig = field(default_factory=PIDConfig)


class LagrangianCMDP:
    """Wrapper tính r_eff = r - Σ λ_i * (cost_i - target_i)_+ và cập nhật λ bằng PID.
    cost_batch là dict[name] -> tensor shape [B].
    """
    def __init__(self, constraints: Dict[str, ConstraintSpec]):
        self.constraints = constraints
        # trạng thái PID
        self.lmbda = {k: torch.tensor(0.0) for k in constraints}
        self.err_int = {k: 0.0 for k in constraints}
        self.err_prev = {k: 0.0 for k in constraints}

    def effective_reward(self, r: torch.Tensor, cost_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        r_eff = r.clone()
        for name, spec in self.constraints.items():
            c = cost_dict[name]
            # lỗi dương nếu vượt target
            e = (c - spec.target).clamp(min=0.0)
            r_eff = r_eff - self.lmbda[name].to(r.device) * e
        return r_eff

    def update_lagrange(self, cost_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        stats = {}
        for name, spec in self.constraints.items():
            c_mean = float(cost_dict[name].mean().item())
            e = max(0.0, c_mean - spec.target)
            # PID
            self.err_int[name] += e
            d = e - self.err_prev[name]
            self.err_prev[name] = e
            u = spec.pid.kp * e + spec.pid.ki * self.err_int[name] + spec.pid.kd * d
            new_lambda = float(self.lmbda[name].item()) + u
            new_lambda = max(spec.pid.clamp_min, min(spec.pid.clamp_max, new_lambda))
            self.lmbda[name] = torch.tensor(new_lambda)
            stats[f"lambda_{name}"] = new_lambda
            stats[f"cost_{name}"] = c_mean
        return stats
