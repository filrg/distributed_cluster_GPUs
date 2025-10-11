import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np


# ---------- Actions ----------
@dataclass(frozen=True)
class RLAction:
    dc_name: str
    n: int
    f: float


# ---------- Metrics (tùy chọn, nếu env cung cấp) ----------
@dataclass(frozen=True)
class StepMetrics:
    # Năng lượng và tải xử lý
    energy_kwh: float = 0.0
    units_processed: float = 0.0  # tổng "unit" đã xử lý trong step
    power_W: float = 0.0          # công suất tức thời trung bình step
    power_state_changes: int = 0  # số lần bật/tắt GPU, DVFS switch, ...

    # Độ trễ
    mean_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class RLEnergyAgentAdv:
    """
    RL tuyến tính theo ngữ cảnh (contextual), nâng cấp:
      - Q(s,a) = w_{job_type}^T phi(s,a) (tách inference/training).
      - Softmax (Boltzmann) trên Q/tau; tau thích ứng theo vi phạm ràng buộc.
      - TD(0) với baseline reward giảm phương sai + clip gradient.
      - Hỗ trợ 2 chế độ reward:
          (1) 'weighted' (multi-objective weighted sum),
          (2) 'constrained' (Lagrangian: tối ưu năng lượng với ràng buộc P99,SLA & Power).
      - Chuẩn hoá đặc trưng (EWMA) để hội tụ ổn định.
    Mặc định vẫn tương thích ngược: nếu không truyền metrics, dùng reward scalar bên ngoài.
    """

    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.0,
                 tau: float = 0.1,
                 eps: float = 0.0,
                 eps_decay: float = 1.0,
                 eps_min: float = 0.0,
                 clip_grad: float = 5.0,
                 baseline_beta: float = 0.01,
                 mode: str = "weighted",     # "weighted" | "constrained"
                 # weighted-sum weights
                 w_energy: float = 1.0,
                 w_intensity: float = 0.5,
                 w_delay: float = 0.05,
                 w_tail: float = 2.0,
                 w_churn: float = 0.01,
                 sla_ms: float = 200.0,
                 # constrained RL (Lagrangian)
                 power_budget_W: Optional[float] = None,
                 lag_lr: float = 0.01,
                 lag_max: float = 1e6,
                 # softmax tau schedule
                 tau_min: float = 0.03,
                 tau_max: float = 0.3,
                 tau_adapt: bool = True,
                 # feature normalizer
                 feat_ewma_beta: float = 0.01,
                 feat_eps: float = 1e-6):
        # Học
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.eps = float(eps)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.clip_grad = float(clip_grad)
        self.baseline_beta = float(baseline_beta)

        # Mục tiêu
        self.mode = str(mode)
        self.w_energy = float(w_energy)
        self.w_intensity = float(w_intensity)
        self.w_delay = float(w_delay)
        self.w_tail = float(w_tail)
        self.w_churn = float(w_churn)
        self.sla_ms = float(sla_ms)

        self.power_budget_W = None if power_budget_W is None else float(power_budget_W)
        self.lag_lr = float(lag_lr)
        self.lag_max = float(lag_max)
        self.lambda_tail = 0.0
        self.lambda_power = 0.0

        # Softmax tau adapt
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_adapt = bool(tau_adapt)

        # Trọng số & baseline tách theo loại job
        self.w_map: Dict[str, np.ndarray] = {}  # "inference"/"training" -> w
        self.rbar: Dict[str, float] = {}        # baseline reward theo loại

        # Chiều đặc trưng và chuẩn hoá
        self._feat_dim = 0
        self._feat_mu: Optional[np.ndarray] = None
        self._feat_var: Optional[np.ndarray] = None
        self._feat_ewma_beta = float(feat_ewma_beta)
        self._feat_eps = float(feat_eps)

    # ---------- Feature engineering ----------
    def _phi_raw(self, state: Dict, action: RLAction) -> np.ndarray:
        """
        State mong đợi (không bắt buộc có hết):
          - job_type: "inference" | "training"
          - job_size
          - dc_total, dc_busy, dc_q_inf, dc_q_trn
          - net_lat
          - (optional per-action):
              t_unit, p_est, e_unit, price_kwh, carbon_g_per_kwh, cap_headroom_W
        Action: f (0..1 hoặc GHz chuẩn hoá), n (#GPU)
        """
        jt_str = "training" if state.get("job_type") == "training" else "inference"
        jt = 1.0 if jt_str == "training" else 0.0

        size = float(state.get("job_size", 0.0))
        size_log = math.log1p(max(0.0, size) * 1e6) / 14.0  # nén theo scale size

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

        # Optional per-action signals
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

        jt_key = jt_str
        if jt_key not in self.w_map:
            self.w_map[jt_key] = np.zeros_like(vec)
            self.rbar[jt_key] = 0.0

        # init stats cho chuẩn hoá
        if self._feat_mu is None:
            self._feat_mu = np.zeros_like(vec)
            self._feat_var = np.ones_like(vec)

        return vec

    def _phi(self, state: Dict, action: RLAction) -> np.ndarray:
        """Chuẩn hoá đặc trưng bằng EWMA mean/var (z-score) để ổn định học."""
        x = self._phi_raw(state, action)

        # cập nhật EWMA mean/var
        beta = self._feat_ewma_beta
        mu = self._feat_mu
        var = self._feat_var
        # online EWMA
        self._feat_mu = (1.0 - beta) * mu + beta * x
        self._feat_var = (1.0 - beta) * var + beta * (x - self._feat_mu) ** 2

        # z-norm
        z = (x - self._feat_mu) / np.sqrt(self._feat_var + self._feat_eps)
        return z

    def _w(self, job_type: str) -> np.ndarray:
        key = "training" if job_type == "training" else "inference"
        if key not in self.w_map:
            dim = self._feat_dim if self._feat_dim > 0 else 1
            self.w_map[key] = np.zeros(dim, dtype=np.float64)
            self.rbar[key] = 0.0
        return self.w_map[key]

    # ---------- Reward models ----------
    def _reward_weighted(self, metrics: StepMetrics) -> float:
        units = max(1e-9, metrics.units_processed)
        energy = metrics.energy_kwh
        energy_intensity = energy / units
        mean_lat = metrics.mean_latency_ms
        p99 = metrics.p99_latency_ms
        sla_violation = max(0.0, p99 - self.sla_ms) / max(self.sla_ms, 1e-9)

        r = (
            - self.w_energy    * energy
            - self.w_intensity * energy_intensity
            - self.w_delay     * (mean_lat / 1000.0)
            - self.w_tail      * sla_violation
            - self.w_churn     * float(metrics.power_state_changes)
        )
        return float(r)

    def _reward_constrained(self, metrics: StepMetrics) -> float:
        """Lagrangian: minimize energy with constraints on P99 and Power."""
        energy = metrics.energy_kwh

        # constraint 1: tail latency
        tail_violation = max(0.0, metrics.p99_latency_ms - self.sla_ms) / max(self.sla_ms, 1e-9)

        # constraint 2: power budget (optional)
        power_violation = 0.0
        if self.power_budget_W is not None and self.power_budget_W > 0:
            power_violation = max(0.0, metrics.power_W - self.power_budget_W) / self.power_budget_W

        # small churn penalty (khuyến nghị giữ nhỏ để tránh nhấp nháy)
        churn_pen = 1e-3 * float(metrics.power_state_changes)

        L = energy + self.lambda_tail * tail_violation + self.lambda_power * power_violation + churn_pen
        return -float(L)

    def _update_lagrange(self, metrics: StepMetrics):
        """Cập nhật bội số Lagrange theo vi phạm (projected gradient ascent)."""
        if self.mode != "constrained":
            return
        # tail
        tail_violation = max(0.0, metrics.p99_latency_ms - self.sla_ms) / max(self.sla_ms, 1e-9)
        self.lambda_tail = min(self.lag_max, max(0.0, self.lambda_tail + self.lag_lr * tail_violation))
        # power
        if self.power_budget_W is not None and self.power_budget_W > 0:
            power_violation = max(0.0, metrics.power_W - self.power_budget_W) / self.power_budget_W
            self.lambda_power = min(self.lag_max, max(0.0, self.lambda_power + self.lag_lr * power_violation))

    def _maybe_adapt_tau(self, metrics: Optional[StepMetrics]):
        if not self.tau_adapt or metrics is None:
            return
        violated = (metrics.p99_latency_ms > self.sla_ms) or (
            self.power_budget_W is not None and metrics.power_W > self.power_budget_W
        )
        # nếu đang vi phạm, tăng nhiệt độ (explore); nếu không, hạ dần tới tau_min (exploit)
        if violated:
            self.tau = min(self.tau_max, self.tau * 1.05 + 1e-3)
        else:
            self.tau = max(self.tau_min, self.tau * 0.995)

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
        tausafe = max(self.tau, 1e-6)
        m = -1e30
        for a in actions:
            s = state_map.get(a.dc_name, {})
            q = self.q_value(s, a)
            z = q / tausafe
            logits.append(z)
            if z > m:
                m = z

        # Ổn định số học
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

    def _compute_internal_reward(
        self,
        reward_scalar: Optional[float],
        metrics: Optional[Union[StepMetrics, Dict]]
    ) -> (float, Optional[StepMetrics]):
        """
        Cho phép 2 cách gọi:
          - (cũ) truyền reward scalar → dùng thẳng.
          - (mới) truyền metrics → tự tính reward theo 'mode'.
        """
        if metrics is None:
            # không có metrics → dùng reward scalar
            if reward_scalar is None:
                raise ValueError("Either reward or metrics must be provided.")
            return float(reward_scalar), None

        # dict -> StepMetrics
        sm = metrics if isinstance(metrics, StepMetrics) else StepMetrics(**metrics)

        if self.mode == "constrained":
            r = self._reward_constrained(sm)
        else:
            r = self._reward_weighted(sm)

        # cập nhật Lagrange & tau theo vi phạm
        self._update_lagrange(sm)
        self._maybe_adapt_tau(sm)

        return float(r), sm

    def update(self,
               state: Dict,
               action: RLAction,
               reward: Optional[float] = None,
               next_state: Optional[Dict] = None,
               next_actions: Optional[List[RLAction]] = None,
               done: bool = False,
               metrics: Optional[Union[StepMetrics, Dict]] = None):
        """
        TD(0) với baseline reward & clip grad.
        - Backward compatible: nếu 'metrics' không được cung cấp → dùng 'reward' như trước.
        - Nếu 'metrics' được cung cấp → agent tự tính reward (weighted/constrained).
        """
        jt = "training" if state.get("job_type") == "training" else "inference"
        phi = self._phi(state, action)
        w = self._w(jt)

        # Reward nội bộ (có thể từ metrics)
        reward_val, sm = self._compute_internal_reward(reward, metrics)

        # Baseline reward để giảm phương sai (moving average theo job_type)
        rb = self.rbar[jt]
        reward_ctr = reward_val - rb
        self.rbar[jt] = (1.0 - self.baseline_beta) * rb + self.baseline_beta * reward_val

        # TD target
        q_sa = float(np.dot(w, phi))
        target = reward_ctr
        if (not done) and next_state is not None and next_actions:
            # dùng max Q ở state kế tiếp (giữ nguyên như cũ; gamma=0 → contextual bandit)
            max_q_next = max(self.q_value(next_state, a) for a in next_actions)
            target += self.gamma * max_q_next
        td = target - q_sa

        # Gradient & clip
        grad = td * phi
        if self.clip_grad > 0:
            norm = float(np.linalg.norm(grad)) + 1e-9
            if norm > self.clip_grad:
                grad = grad * (self.clip_grad / norm)

        self.w_map[jt] = w + self.alpha * grad

        # Epsilon decay
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
