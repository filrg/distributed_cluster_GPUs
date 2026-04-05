import numpy as np
import torch
from typing import Any, List, Optional, Tuple

from simcore.models.datacenter.datacenter import DataCenter
from simcore.models.job import Job, PreemptedJob
from ..policy_paper import best_energy_freq
from simcore.rl.chsac_af import CHSAC_AF
from simcore.rl.replay import ReplayBuffer, Transition


class ChsacAFRuntime:
    """Encapsulate CHSAC-AF specific state, decisions, and training updates for the simulator."""

    def __init__(
        self,
        simulator: Any,
        sla_p99_ms: float,
        upgr_batch: int,
        upgr_warmup: int,
        upgr_buffer: int,
    ):
        self.sim = simulator
        self._n_g_choices = int(self.sim.policy.max_gpus_per_job)

        f_all = [f for dc in self.sim.dcs.values() for f in getattr(dc, "freq_levels", [])]
        self._f_min = min(f_all) if f_all else 1.0
        self._obs_dim = self.obs().shape[0]

        constraints = {"latency_p99": sla_p99_ms, "gpu_over": 0.0}
        if self.sim.power_cap and self.sim.power_cap > 0:
            constraints["power"] = float(self.sim.power_cap)
        if self.sim.energy_budget_j and self.sim.energy_budget_j > 0:
            constraints["energy_total"] = float(self.sim.energy_budget_j)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rl_upgr = CHSAC_AF(
            obs_dim=self._obs_dim,
            n_dc=len(self.sim._dc_names),
            n_g_choices=self._n_g_choices,
            constraints=constraints,
            device=device,
        )
        self._upgr_batch = upgr_batch
        self._upgr_warmup = upgr_warmup
        self._replay = ReplayBuffer(capacity=upgr_buffer, state_dim=self._obs_dim, device=device)

    def obs(self) -> np.ndarray:
        """Vector hoá trạng thái cho RL Upgr: [now] + cho từng DC: [total, busy, free, current_f, qlen_inf, qlen_trn]."""
        feats = []
        for dc in self.sim.dcs.values():
            total = float(getattr(dc, "total_gpus", dc.total_gpus))
            busy = float(dc.busy_gpus)
            free = max(0.0, total - busy)
            cf = float(getattr(dc, "current_freq", self._f_min))
            q_inf = float(len(dc.q_inf))
            q_trn = float(len(dc.q_train))
            feats.extend([total, busy, free, cf, q_inf, q_trn])
        now = float(self.sim.now)
        return np.asarray([now] + feats, dtype=np.float32)

    def masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mask cho DC (có GPU rảnh) và cho #GPU (1..N) theo free tối đa trên mọi DC."""
        dc_mask = []
        max_free = 0
        for dc in self.sim.dcs.values():
            total = int(getattr(dc, "total_gpus", dc.total_gpus))
            free = max(0, total - int(dc.busy_gpus))
            dc_mask.append(free > 0)
            max_free = max(max_free, free)

        g_mask = [(i + 1) <= max_free for i in range(self._n_g_choices)]

        buf = None
        if hasattr(self.sim, "_lat_hist"):
            buf = self.sim._lat_hist.get("training") or self.sim._lat_hist.get("inference")

        if buf and len(buf) >= 5:
            p99_recent_ms = float(np.percentile(buf, 99) * 1000.0)
            target = self.rl_upgr.cmdp.constraints["latency_p99"].target
            if p99_recent_ms < 0.9 * target:
                cap = 1
                g_mask = np.array([(i + 1) <= min(cap, max_free) for i in range(self._n_g_choices)], dtype=bool)

        return np.asarray(dc_mask, bool), np.asarray(g_mask, bool)

    def _select_action(self):
        obs = self.obs()
        m_dc, m_g = self.masks()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        m_dc_t = torch.from_numpy(m_dc).unsqueeze(0).bool()
        m_g_t = torch.from_numpy(m_g).unsqueeze(0).bool()
        action = self.rl_upgr.select_action(obs_t, m_dc_t, m_g_t, deterministic=False)
        return obs, action

    def route_arrival(self, job: Job) -> str:
        obs, action = self._select_action()
        dc_name = self.sim._dc_idx_to_name(action["dc"])
        n_sel = self.sim._gidx_to_n(action["g"])
        job._upgr_state0 = obs
        job._upgr_action = {"dc_idx": int(action["dc"]), "g_idx": int(action["g"]), "n": n_sel}
        return dc_name

    def fixed_dc_allocation(self, dc: DataCenter, job: Job) -> Optional[Tuple[int, float]]:
        if not hasattr(job, "_upgr_action"):
            return None
        a = job._upgr_action
        n = max(1, min(a["n"], dc.free_gpus, self.sim.policy.max_gpus_per_job))
        pC, tC = self.sim.coeffs_map[(dc.name, job.jtype)]
        levels = sorted(dc.freq_levels) if getattr(dc, "freq_levels", None) else [getattr(dc, "current_freq", 1.0)]
        f_opt = best_energy_freq(n, levels, pC, tC)
        f_opt = self.sim._guard_freq_by_deadline(job, n, levels, tC, f_opt)
        return n, float(f_opt)

    def queue_allocation(self, job: Job):
        obs, action = self._select_action()
        dc_tgt = self.sim.dcs[self.sim._dc_idx_to_name(action["dc"])]
        if dc_tgt.free_gpus <= 0:
            return None

        n_sel = max(1, min(self.sim._gidx_to_n(int(action["g"])), dc_tgt.free_gpus, self.sim.policy.max_gpus_per_job))
        pC, tC = self.sim.coeffs_map[(dc_tgt.name, job.jtype)]
        levels = sorted(dc_tgt.freq_levels)
        f_opt = best_energy_freq(n_sel, levels, pC, tC)
        f_sel = self.sim._guard_freq_by_deadline(job, n_sel, levels, tC, f_opt)

        job._upgr_state0 = obs
        job._upgr_action = {"dc_idx": int(action["dc"]), "g_idx": int(action["g"]), "n": n_sel}
        return dc_tgt, n_sel, float(f_sel)

    def reallocate_training_jobs(self, dc: DataCenter, preempted_jobs: List[PreemptedJob]):
        for preempted_job in preempted_jobs:
            job = preempted_job.job
            _, action = self._select_action()
            n_rl = self.sim._gidx_to_n(int(action["g"]))
            n_rl = max(1, min(n_rl, dc.free_gpus, self.sim.policy.max_gpus_per_job))
            pC, tC = self.sim.coeffs_map[(dc.name, job.jtype)]
            levels = sorted(dc.freq_levels) if dc.freq_levels else [getattr(dc, "current_freq", 1.0)]
            f_rl = best_energy_freq(n_rl, levels, pC, tC)
            f_rl = self.sim._guard_freq_by_deadline(job, n_rl, levels, tC, f_rl)
            self.sim._resume_preempted_job(dc, preempted_job, n_rl, f_rl)

    def on_job_finish(self, dc: DataCenter, job: Job, g: int, f_used: float, rl_metrics: dict):
        if not (hasattr(job, "_upgr_action") and hasattr(job, "_upgr_state0")):
            return

        e_unit_kwh = float(rl_metrics["energy_kwh"]) / (float(rl_metrics["units_processed"]) + 1e-9)
        n = max(1, int(job._upgr_action["n"]))
        alpha_n = 0.05
        gpu_pref = alpha_n * (1.0 / n)
        reward = -e_unit_kwh + gpu_pref

        sla_ms_target = self.rl_upgr.cmdp.constraints["latency_p99"].target
        n_min = self.sim._min_n_for_sla(dc, job, f_used, sla_ms_target)
        gpu_over = max(0, g - n_min)

        costs = {
            "latency_p99": float(rl_metrics["p99_latency_ms"]),
            "power": float(rl_metrics["power_W"]),
            "gpu_over": float(gpu_over),
        }

        s0 = job._upgr_state0
        s1 = self.obs()
        a_dc_idx = int(job._upgr_action["dc_idx"])
        a_g_idx = int(job._upgr_action["g_idx"])
        m_dc, m_g = self.masks()

        self._replay.add(Transition(
            s=s0,
            s_next=s1,
            a_dc=a_dc_idx,
            a_g=a_g_idx,
            r=reward,
            costs=costs,
            done=True,
            mask_dc=m_dc,
            mask_g=m_g,
        ))

        if self._replay.size >= self._upgr_warmup:
            batch = self._replay.sample(self._upgr_batch)
            stats = self.rl_upgr.train_step(batch)
            if stats:
                self.sim.logger.info({k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in stats.items()})

        del job._upgr_state0
        del job._upgr_action
