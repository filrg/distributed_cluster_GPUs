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


class RLEnergyAgentAdv:
    """
    RL tuyến tính theo ngữ cảnh (contextual), ổn định hơn:
      - Q(s,a) = w_{job_type}^T phi(s,a), tách trọng số cho inference/training.
      - Chọn hành động bằng softmax (Boltzmann) trên Q/tau (ổn định hơn epsilon-greedy).
      - Update TD(0) với baseline reward để giảm phương sai + clip gradient.
    Kỳ vọng dùng γ=0 (contextual bandit). Khi muốn học dài hạn, tăng γ lên nhẹ sau.
    """

    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.0,
                 tau: float = 0.1,
                 eps: float = 0.0,
                 eps_decay: float = 1.0,
                 eps_min: float = 0.0,
                 clip_grad: float = 5.0,
                 baseline_beta: float = 0.01):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.eps = float(eps)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.clip_grad = float(clip_grad)
        self.baseline_beta = float(baseline_beta)

        # Trọng số & baseline tách theo loại job
        self.w_map: Dict[str, np.ndarray] = {}  # "inference"/"training" -> w
        self.rbar: Dict[str, float] = {}  # baseline reward theo loại

        # Lưu số chiều đặc trưng để init w đúng kích thước
        self._feat_dim = 0

    # ---------- Feature engineering ----------
    def _phi(self, state: Dict, action: RLAction) -> np.ndarray:
        """
        State mong đợi (nhưng không bắt buộc phải đủ hết):
          - job_type: "inference" | "training"
          - job_size
          - dc_total, dc_busy, dc_q_inf, dc_q_trn
          - net_lat
          - (tùy chọn, nếu simulator đã tính per-action & inject vào state):
              t_unit, p_est, e_unit, price_kwh, carbon_g_per_kwh, cap_headroom_W
        Action:
          - f (đã chuẩn hoá 0..1), n (GPU)
        """
        jt_str = "training" if state.get("job_type") == "training" else "inference"
        jt = 1.0 if jt_str == "training" else 0.0

        size = float(state.get("job_size", 0.0))
        size_log = math.log1p(max(0.0, size) * 1e6) / 14.0  # nén tuỳ scale đơn vị "size"

        total = float(state.get("dc_total", 1.0))
        busy = float(state.get("dc_busy", 0.0))
        util = busy / total if total > 0 else 0.0

        q_inf = float(state.get("dc_q_inf", 0.0))
        q_trn = float(state.get("dc_q_trn", 0.0))
        q_inf_n = math.tanh(q_inf / 16.0)
        q_trn_n = math.tanh(q_trn / 16.0)

        net_lat_n = math.tanh(float(state.get("net_lat", 0.0)) / 0.2)  # 0.2s scale

        f_norm = float(action.f)
        n_norm = float(action.n) / max(1.0, total)

        # Optional per-action signals (nên có để hội tụ nhanh theo mô hình năng lượng)
        t_unit = float(state.get("t_unit", 0.0))
        p_est = float(state.get("p_est", 0.0))
        e_unit = float(state.get("e_unit", 0.0))
        t_unit_n = math.tanh(t_unit / 0.05)
        p_est_n = math.tanh(p_est / 2000.0)
        e_unit_n = math.tanh(e_unit / 100.0)

        price_kwh = float(state.get("price_kwh", 0.0))
        carbon_k = float(state.get("carbon_g_per_kwh", 0.0)) / 1000.0  # kg/kWh
        cap_headroom = float(state.get("cap_headroom_W", 0.0))
        cap_headroom_n = math.tanh(cap_headroom / 2000.0)

        vec = np.array([
            1.0,  # bias
            jt, size_log,
            util, q_inf_n, q_trn_n,
            net_lat_n,
            f_norm, n_norm,
            t_unit_n, p_est_n, e_unit_n,
            price_kwh, carbon_k, cap_headroom_n
        ], dtype=np.float64)

        # init w theo đúng chiều đặc trưng
        if self._feat_dim == 0:
            self._feat_dim = vec.size
        elif vec.size != self._feat_dim:
            # an toàn: pad/cắt về cùng chiều
            if vec.size < self._feat_dim:
                vec = np.pad(vec, (0, self._feat_dim - vec.size))
            else:
                vec = vec[:self._feat_dim]

        if jt_str not in self.w_map:
            self.w_map[jt_str] = np.zeros_like(vec)
            self.rbar[jt_str] = 0.0

        return vec

    def _w(self, job_type: str) -> np.ndarray:
        key = "training" if job_type == "training" else "inference"
        if key not in self.w_map:
            self.w_map[key] = np.zeros(self._feat_dim if self._feat_dim > 0 else 1, dtype=np.float64)
            self.rbar[key] = 0.0
        return self.w_map[key]

    # ---------- API ----------
    def q_value(self, state: Dict, action: RLAction) -> float:
        jt = "training" if state.get("job_type") == "training" else "inference"
        phi = self._phi(state, action)
        return float(np.dot(self._w(jt), phi))

    def select_per_dc(self, state_map: Dict[str, Dict], actions: List[RLAction]) -> RLAction:
        """
        Chọn hành động bằng softmax trên Q/tau; thỉnh thoảng random theo eps để khám phá.
        state_map: dc_name -> state dict (đã augment theo DC)
        """
        if not actions:
            raise ValueError("No actions")

        # Epsilon thăm dò (hiếm)
        if self.eps > 0.0 and random.random() < self.eps:
            return random.choice(actions)

        # Softmax
        logits = []
        for a in actions:
            s = state_map.get(a.dc_name, {})
            q = self.q_value(s, a)
            logits.append(q / max(self.tau, 1e-6))

        # Ổn định số học
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        Z = sum(exps)
        if not math.isfinite(Z) or Z <= 0.0:
            return random.choice(actions)

        r = random.random() * Z
        acc = 0.0
        for a, e in zip(actions, exps):
            acc += e
            if r <= acc:
                return a
        return actions[-1]

    def update(self, state: Dict, action: RLAction, reward: float,
               next_state: Dict, next_actions: List[RLAction], done: bool):
        """
        TD(0) với baseline reward & clip grad. Với gamma=0 → contextual bandit.
        """
        jt = "training" if state.get("job_type") == "training" else "inference"
        phi = self._phi(state, action)
        w = self._w(jt)

        # Baseline reward để giảm phương sai (moving average)
        rb = self.rbar[jt]
        reward_ctr = reward - rb
        self.rbar[jt] = (1.0 - self.baseline_beta) * rb + self.baseline_beta * reward

        q_sa = float(np.dot(w, phi))
        target = reward_ctr
        if (not done) and next_actions:
            max_q_next = max(self.q_value(next_state, a) for a in next_actions)
            target += self.gamma * max_q_next
        td = target - q_sa

        grad = td * phi
        if self.clip_grad > 0:
            norm = float(np.linalg.norm(grad)) + 1e-9
            if norm > self.clip_grad:
                grad = grad * (self.clip_grad / norm)

        self.w_map[jt] = w + self.alpha * grad

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
