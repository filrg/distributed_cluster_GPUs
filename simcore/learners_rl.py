import math
import random
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass(frozen=True)
class RLAction:
    dc_name: str
    n: int
    f: float


class RLEnergyAgent:
    """
    Q-learning tuyến tính: Q(s,a) = w^T phi(s,a).
    - select(): epsilon-greedy trên tập hành động rời rạc {(dc, n, f)}
    - update(): TD(0) với mục tiêu r + γ max_a' Q(s',a')
    """
    def __init__(self, alpha=0.1, gamma=0.0, eps=0.2, eps_decay=0.995, eps_min=0.02):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.w = None  # vector trọng số (numpy)

    # -------- Feature engineering đơn giản, ổn định scale --------
    def _phi(self, state: Dict, action: RLAction) -> np.ndarray:
        """
        state gồm:
          - job_type: 0/1
          - job_size
          - dc_total, dc_busy, dc_q_inf, dc_q_trn
          - net_lat (ước lượng)
          - f_levels_len (để scale)
        action:
          - one-hot 'is_this_dc' (đã encode bằng cách chỉ trích đặc trưng của DC được chọn)
          - f_norm, n_norm
        """
        jt = 1.0 if state.get("job_type") == "training" else 0.0
        size = float(state.get("job_size", 0.0))
        size_log = math.log1p(size * 1e6) / 14.0  # nén [tuỳ vào đơn vị unit]
        total = float(state.get("dc_total", 1.0))
        busy = float(state.get("dc_busy", 0.0))
        q_inf = float(state.get("dc_q_inf", 0.0))
        q_trn = float(state.get("dc_q_trn", 0.0))
        util = busy / total if total > 0 else 0.0
        q_inf_n = math.tanh(q_inf / 16.0)
        q_trn_n = math.tanh(q_trn / 16.0)
        net_lat = float(state.get("net_lat", 0.0))
        net_lat_n = math.tanh(net_lat / 0.2)  # 200ms scale

        # action features
        f_levels_len = max(1, int(state.get("f_levels_len", 1)))
        # giả sử f được chuẩn hoá 0..1 sẵn
        f_norm = float(action.f)
        n_norm = float(action.n) / max(1.0, total)

        vec = np.array([
            1.0,  # bias
            jt, size_log,
            util, q_inf_n, q_trn_n,
            net_lat_n,
            f_norm, n_norm
        ], dtype=np.float64)
        if self.w is None:
            self.w = np.zeros_like(vec)
        return vec

    def q_value(self, state: Dict, action: RLAction) -> float:
        phi = self._phi(state, action)
        return float(np.dot(self.w, phi))

    def select_per_dc(self, state_map: Dict[str, Dict], actions: List[RLAction]) -> RLAction:
        """
        Chọn epsilon-greedy với state phụ thuộc DC:
        state_map: dc_name -> state dict tương ứng cho action dc_name.
        """
        if not actions:
            raise ValueError("No actions")
        # khám phá
        if (self.w is None) or (random.random() < self.eps):
            return random.choice(actions)
        # khai thác
        best_a, best_q = None, -1e30
        for a in actions:
            s = state_map.get(a.dc_name, {})
            q = self.q_value(s, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update(self, state: Dict, action: RLAction, reward: float, next_state: Dict, next_actions: List[RLAction],
               done: bool):
        phi = self._phi(state, action)
        q_sa = float(np.dot(self.w, phi))
        target = reward
        if (not done) and next_actions:
            max_q_next = max(self.q_value(next_state, a) for a in next_actions)
            target += self.gamma * max_q_next
        td = target - q_sa
        self.w += self.alpha * td * phi
        # decay epsilon nhẹ
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
