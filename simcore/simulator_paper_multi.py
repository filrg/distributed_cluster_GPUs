import csv, heapq, itertools, random, os, numpy as np, torch
from typing import Dict, List, Tuple, Optional, Union
from .models import Job, DataCenter, PreemptedJob
from .arrivals import ArrivalConfig, sample_job_size
from .policy import PolicyConfig, select_gpus_and_set_freq
from .latency_paper import step_time_s
from .policy_paper import energy_tuple, best_nf_grid, best_energy_freq
from .network import Ingress, Graph
from .router import RouterPolicy
from .energy_paper import task_power_w
from .learners import BanditDVFS
from .freq_load_agg import TaskState, aggregate_with_atoms
from simcore.rl.rl_energy_agent_adv_upgrade import RLEnergyAgentAdvUpgr
from simcore.rl.replay import ReplayBuffer, Transition

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

Event = Tuple[float, int, str, dict]


class MultiIngressPaperSimulator:
    """
    - Mỗi ingress có arrival riêng (inference & training).
    - Khi job đến ingress: route -> DC, cộng delay WAN rồi mới vào hàng đợi DC.
    - Data transfer time (đơn giản): t_net = Lnet + data_GB / bottleneck_Gbps (nếu biết).
    """

    def __init__(self,
                 ingresses: Dict[str, Ingress],
                 dcs: Dict[str, DataCenter],
                 graph: Graph,
                 arrival_inf: ArrivalConfig,
                 arrival_train: ArrivalConfig,
                 router_policy: RouterPolicy,
                 coeffs_map: Dict[Tuple, Tuple],
                 logger,
                 carbon_intensity: Optional[Dict[str, float]] = None,
                 energy_price: Optional[Union[Dict[int, float], Dict[str, Dict[int, float]]]] = None,
                 policy: PolicyConfig = None,
                 sim_duration: float = 3600.0,
                 log_interval: float = 10.0,
                 log_path: str = None,
                 rng_seed: int = 42,
                 algo: str = "baseline",
                 elastic_scaling: bool = False,
                 power_cap: float = 0.0,
                 energy_budget_j: float = 0.0,
                 sla_p99_ms: float = 500.0,
                 control_interval: float = 5.0,
                 show_progress: bool = True,
                 upgr_batch: int = 256, upgr_warmup: int = 1000, upgr_buffer: int = 200000,
                 num_fixed_gpus=1, fixed_freq=None):
        self.now = 0.0
        self.end_time = sim_duration
        self.event_q: List[Event] = []
        self.seq = itertools.count()
        self.ingresses = ingresses
        self.dcs = dcs
        self.graph = graph
        self.arr_inf = arrival_inf
        self.arr_trn = arrival_train
        self.router_policy = router_policy
        self.coeffs_map = coeffs_map
        self.carbon = carbon_intensity or {}
        self.energy_price = energy_price or {}
        self.policy = policy or PolicyConfig(name="energy_aware")
        self.logger = logger
        random.seed(rng_seed)
        self.jid_counter = itertools.count(1)

        self.cluster_log_path = "cluster_log.csv"
        self.job_log_path = "job_log.csv"
        if log_path:
            os.makedirs(log_path, exist_ok=True)
            self.cluster_log_path = os.path.join(log_path, "cluster_log.csv")
            self.job_log_path = os.path.join(log_path, "job_log.csv")

        self.algo = algo
        self.elastic_scaling = elastic_scaling and self.algo in ("rl_energy_upgr")
        # self.rl = None
        # self.rl_n_cand = int(rl_n_cand)

        self.power_cap = power_cap
        self.energy_budget_j = energy_budget_j
        self.control_interval = control_interval
        self.bandit = BanditDVFS(init_explore=1, objective="energy") if algo == "bandit" else None

        # === RL Upgraded mode ===
        self.rl_upgr = None
        if self.algo == "rl_energy_upgr":
            # danh sách DC cố định để map chỉ số <-> tên
            self._dc_names: List[str] = list(self.dcs.keys())

            # bắc cầu số lựa chọn GPU: 1..N (dựa theo max_gpus_per_job hoặc max free)
            self._n_g_choices = int(self.policy.max_gpus_per_job)

            # biên DVFS toàn cục (lấy min/max trên tất cả DC có freq_levels)
            f_all = [f for dc in self.dcs.values() for f in getattr(dc, "freq_levels", [])]
            self._f_min = min(f_all) if f_all else 1.0

            # kích thước quan sát (thêm helper bên dưới)
            self._obs_dim = self._upgr_obs().shape[0]

            # Constraints mặc định: dùng SLA inference nếu có, + power_cap (nếu >0)
            constraints = {}
            constraints["latency_p99"] = sla_p99_ms
            if self.power_cap and self.power_cap > 0:
                constraints["power"] = float(self.power_cap)
            if self.energy_budget_j and self.energy_budget_j > 0:
                constraints["energy_total"] = float(self.energy_budget_j)
            constraints["gpu_over"] = 0.0

            self.rl_upgr = RLEnergyAgentAdvUpgr(
                obs_dim=self._obs_dim,
                n_dc=len(self._dc_names),
                n_g_choices=self._n_g_choices,
                constraints=constraints,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Replay buffer và hyper đơn giản
            self._upgr_batch = upgr_batch
            self._upgr_warmup = upgr_warmup
            self._replay = ReplayBuffer(capacity=upgr_buffer, state_dim=self._obs_dim,
                                        device="cuda" if torch.cuda.is_available() else "cpu")

        # debug
        self.num_fixed_gpus = num_fixed_gpus
        self.fixed_freq = fixed_freq

        self.log_interval = log_interval
        self.show_progress = bool(show_progress)
        self._pbar = None
        self._pbar_last_t = 0.0
        if self.show_progress and tqdm is not None:
            self._pbar = tqdm(
                total=self.end_time,
                desc="Sim time",
                unit="s",
                dynamic_ncols=True,
                mininterval=0.2,  # hạn chế spam stdout
                smoothing=0.3,
                bar_format=("{desc}: {percentage:3.0f}%|{bar}| "
                            "{n:,.0f}/{total:,.0f} [{elapsed}<{remaining}, {rate_fmt}]")
            )
        elif self.show_progress and tqdm is None:
            print("[info] tqdm chưa có. Cài: pip install tqdm (đã tự tắt progress).")
            self.show_progress = False

        # schedule arrivals per ingress
        for ing in self.ingresses.values():
            self._schedule(self.now + self.arr_inf.next_interarrival(self.now), 'arrival_inf', {'ing': ing.name})
            self._schedule(self.now + self.arr_trn.next_interarrival(self.now), 'arrival_trn', {'ing': ing.name})
        self._schedule(self.now + log_interval, 'log', {'interval': log_interval})

    # --- event helpers ---
    def _schedule(self, t: float, etype: str, payload: dict):
        if t == float('inf') or t > self.end_time + 1e-9:
            return
        heapq.heappush(self.event_q, (t, next(self.seq), etype, payload))

    def _pop(self):
        return heapq.heappop(self.event_q) if self.event_q else None

    def _estimate_dc_power(self, dc: DataCenter, f: float) -> float:
        # Active (paper model)
        p_active = 0.0
        for job, g in dc.running_jobs.values():
            pC, tC = self.coeffs_map[(dc.name, job.jtype)]
            f_job = getattr(job, "f_used", f)
            p_active += task_power_w(g, f_job, pC)
        # Idle
        idle = dc.total_gpus - dc.busy_gpus
        gt = dc.gpu_type
        p_idle = idle * (gt.p_sleep if dc.power_gating else gt.p_idle)
        return p_active + p_idle

    def _cap_uniform(self, deficit: float):
        """Giảm f theo bước rời rạc, mỗi bước chọn DC có ∆P lớn nhất."""
        iter_guard = 10000
        while deficit > 1e-6 and iter_guard > 0:
            iter_guard -= 1
            best_dc, best_dp, best_next_f = None, 0.0, None
            for dc in self.dcs.values():
                levels = dc.freq_levels
                # snap index
                try:
                    idx = levels.index(dc.current_freq)
                except ValueError:
                    idx = min(range(len(levels)), key=lambda i: abs(levels[i] - dc.current_freq))
                if idx == 0:
                    continue
                f_now, f_lo = levels[idx], levels[idx - 1]
                p_now = self._estimate_dc_power(dc, f_now)
                p_lo = self._estimate_dc_power(dc, f_lo)
                dp = p_now - p_lo
                if dp > best_dp + 1e-9:
                    best_dc, best_dp, best_next_f = dc, dp, f_lo
            if not best_dc or best_dp <= 1e-9:
                break
            best_dc.current_freq = best_next_f
            deficit -= best_dp

    def _control(self):
        """
        Power-cap controller:
          - cap_uniform: hạ đồng loạt theo DC, mỗi bước chọn DC có ∆P lớn nhất.
          - cap_greedy : hạ theo 'atoms' per-job (từ aggregate), tái-ước-lượng sau mỗi bước.
        Có hysteresis nhỏ để tránh rung quanh power_cap.
        """
        # Chỉ chạy khi cần
        if self.power_cap <= 0:
            return

        algo = getattr(self, "algo", "baseline")
        if algo not in ("cap_uniform", "cap_greedy"):
            # (tuỳ chọn) vài heuristic nhẹ cho các algo khác
            if algo in ("eco_route", "carbon_cost"):
                # hạ f về min khi DC rảnh để giảm idle power
                for dc in self.dcs.values():
                    if dc.busy_gpus == 0 and dc.freq_levels:
                        dc.current_freq = min(dc.freq_levels)
            return

        # 1) Ước lượng tổng P hiện tại
        totalP = 0.0
        for dc in self.dcs.values():
            f_now = getattr(dc, "current_freq", 1.0)
            totalP += self._estimate_dc_power(dc, f_now)

        # Hysteresis: nếu đã dưới cap một chút thì thôi
        cap_margin = getattr(self, "cap_margin", 5.0)  # W
        if totalP <= self.power_cap - cap_margin:
            return

        # Thâm hụt cần cắt
        deficit = max(0.0, totalP - self.power_cap)
        if deficit <= 1e-6:
            return

        # 2) Nhánh cap_uniform: gọi hàm sẵn có
        if algo == "cap_uniform":
            return self._cap_uniform(deficit)

        iter_guard = 10000  # van an toàn, tránh vòng vô tận
        while deficit > 1e-6 and iter_guard > 0:
            iter_guard -= 1

            # 3.1 Gom tasks (bỏ qua DC không có freq_levels hoặc job đã ở f min)
            tasks = []
            for dc in self.dcs.values():
                levels = getattr(dc, "freq_levels", None) or []
                if not levels:
                    continue
                f_min = min(levels)
                for job, g in list(dc.running_jobs.values()):
                    pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                    cur_f = getattr(job, "f_used", 0.0) or dc.current_freq
                    if cur_f <= f_min + 1e-12:
                        continue  # không còn nấc để hạ
                    tasks.append(TaskState(job_id=job.jid, dc_name=dc.name,
                                           n=g, f=cur_f, freq_levels=levels,
                                           p_coeffs=pC, t_coeffs=tC))

            if not tasks:
                break

            # 3.2 Lấy danh sách atoms "đi xuống" đã sắp xếp theo ưu tiên (∆P/∆V) từ aggregate
            try:
                _, down_atoms = aggregate_with_atoms(tasks)
            except Exception:
                break
            if not down_atoms:
                break

            applied_any = False

            # 3.3 Áp dụng lần lượt atoms cho tới khi hết deficit
            for atom in down_atoms:
                if deficit <= 1e-6:
                    break

                dc = self.dcs.get(atom.dc_name)
                if dc is None:
                    continue
                tup = dc.running_jobs.get(atom.job_id)
                if not tup:
                    continue

                job, g = tup
                cur_f = getattr(job, "f_used", dc.current_freq)
                # Bỏ qua atoms "không đi xuống"
                if atom.f_to >= cur_f - 1e-12:
                    continue

                # Thực hiện reschedule về f_to
                self._reschedule_job(dc, job, g, atom.f_to)
                applied_any = True

                # 3.4 Tái-ước-lượng totalP CHÍNH XÁC sau khi đổi f
                new_totalP = 0.0
                for _dc in self.dcs.values():
                    new_totalP += self._estimate_dc_power(_dc, getattr(_dc, "current_freq", 1.0))
                totalP = new_totalP
                deficit = max(0.0, totalP - self.power_cap)

                if deficit <= 1e-6:
                    break

            # Nếu vòng này không áp dụng được atom nào → thoát
            if not applied_any:
                break

    def _job_rate_units_per_s(self, dc, job, gpus, f):
        # 1 unit công việc mất T_unit(n,f) giây ⇒ tốc độ (units/s) = 1 / T_unit
        _, tC = self.coeffs_map[(dc.name, job.jtype)]
        T_unit = step_time_s(gpus, f, tC)
        return 1.0 / max(T_unit, 1e-9)

    def _advance_progress_to_now(self, dc, job, gpus):
        # cộng dồn units_done theo f_used hiện tại
        rate = self._job_rate_units_per_s(dc, job, gpus, job.f_used or dc.current_freq)
        dt = max(0.0, self.now - job.last_update)
        job.units_done = min(job.units_total, job.units_done + rate * dt)
        job.last_update = self.now

    def _reschedule_job(self, dc, job, gpus, new_f):
        # cập nhật tiến độ -> đổi f -> đặt lại job_finish
        self._advance_progress_to_now(dc, job, gpus)
        job.f_used = new_f
        units_left = max(0.0, job.units_total - job.units_done)
        rate_new = self._job_rate_units_per_s(dc, job, gpus, new_f)
        finish_in = units_left / max(rate_new, 1e-9)
        job.ev_gen += 1
        self._schedule(self.now + finish_in, 'job_finish', {'dc': dc.name, 'jid': job.jid, 'gen': job.ev_gen})

    def _preempt_job(self, dc: DataCenter, job: Job, reason: str):
        if job.jid not in dc.running_jobs:
            return
        job, gpus = dc.running_jobs.pop(job.jid)
        self._advance_progress_to_now(dc, job, gpus)

        preempt_ckpt = {
            "units_done": job.units_done,
            "f_used": job.f_used,
            "gpus_assigned": gpus,
        }
        preempted = PreemptedJob(
            job=job,
            preempt_time=self.now,
            reason=reason,
            preempt_ckpt=preempt_ckpt
        )
        dc.preempted_jobs.append(preempted)

        dc.busy_gpus -= gpus
        job.preempt_count += 1

    def _resume_preempted_job(self, dc: DataCenter, preempted: PreemptedJob, n_resume, f_resume):
        job = preempted.job
        if dc.free_gpus < n_resume:
            return False
        # Ckpt restore
        job.units_done = preempted.preempt_ckpt["units_done"]
        job.f_used = f_resume # job.f_used = preempted.preempt_ckpt["f_used"]
        job.gpus_assigned = n_resume
        job.last_update = self.now

        preempt_duration = self.now - preempted.preempt_time
        job.total_preempt_time += preempt_duration
        # Resume
        dc.busy_gpus += n_resume
        dc.running_jobs[job.jid] = (job, n_resume)

        units_left = max(0.0, job.units_total - job.units_done)
        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        T_unit = step_time_s(n_resume, f_resume, t_coeffs)
        finish_time = units_left / max(1.0 / T_unit, 1e-9)

        job.ev_gen += 1
        self._schedule(self.now + finish_time, "job_finish",
                       {'dc': dc.name, 'jid': job.jid, 'gen': job.ev_gen})
        dc.preempted_jobs.remove(preempted)
        self.logger.info(f"Resume job {job.jid} at {self._current_hour()}.")

    def _should_reallocation(self, dc: DataCenter, job_type: str) -> bool:
        """Re-allocate on train completion (only with elastic_scaling)"""
        if job_type != "training":
            return False
        num_running_train_jobs = sum(1 for job, _ in dc.running_jobs.values() if job.jtype == "training")
        return self.elastic_scaling and num_running_train_jobs > 1

    def _preempt_all_training_jobs(self, dc: DataCenter, reason: str) -> List[PreemptedJob]:
        """Preempted all training jobs and return list of it"""
        preempted_jobs = []
        running_train_jobs = []
        # find all running training jobs
        for jid, (job, gpus) in list(dc.running_jobs.items()):
            if job.jtype == "training":
                running_train_jobs.append((jid, job, gpus))
        # preempt one by one
        for jid, job, gpus in running_train_jobs:
            self._preempt_job(dc, job, reason)
            preempted = next((p for p in dc.preempted_jobs if p.job.jid == jid), None)
            if preempted: preempted_jobs.append(preempted)
        return preempted_jobs

    # --- run loop ---
    def run(self):
        with open(self.cluster_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["time_s", "dc", "freq", "busy", "free",
                                    "run_total", "run_inf", "run_train",
                                    "q_inf", "q_train",
                                    "util_inst", "util_avg", "acc_job_unit",
                                    "power_W", "energy_kJ"])
        with open(self.job_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["jid", "ingress", "type", "size", "dc", "f_used", "n_gpus",
                                    "net_lat_s", "start_s", "finish_s", "latency_s", "preempt_count", "T_pred", "P_pred", "E_pred"])

        while self.event_q:
            ev = self._pop()
            if ev is None: break
            t, _, etype, payload = ev
            if t > self.end_time: break

            for dc in self.dcs.values():
                if dc.util_last_ts == 0.0:
                    dc.util_last_ts = t
                    dc.util_begin_ts = t
                else:
                    dt = max(0.0, t - dc.util_last_ts)
                    dc.util_gpu_time += dc.busy_gpus * dt
                    dc.util_last_ts = t
                dc.accrue_energy(t, power_fn=lambda d, f=dc.current_freq: self._estimate_dc_power(d, f))

            # update progress bar with "simulated time advanced"
            if self._pbar is not None:
                delta = max(0.0, t - self._pbar_last_t)
                if delta > 0.0:
                    self._pbar.update(delta)
                    self._pbar_last_t = t

            self.now = t
            if etype == 'arrival_inf':
                self._handle_ingress_arrival('inference', payload['ing'])
            elif etype == 'arrival_trn':
                self._handle_ingress_arrival('training', payload['ing'])
            elif etype == 'xfer_done':
                self._handle_transfer_done(payload)
            elif etype == 'job_finish':
                # self._handle_job_finish(payload['dc'], payload['jid'])
                dc = self.dcs[payload['dc']]
                tup = dc.running_jobs.get(payload['jid'])
                if not tup:
                    continue
                job, g = tup
                if payload.get('gen') != job.ev_gen:
                    continue  # event cũ, bỏ qua
                self._handle_job_finish(payload['dc'], payload['jid'])
            elif etype == 'log':
                self._control()
                self._handle_log(payload['interval'])
            else:
                raise RuntimeError(f"Unknown event {etype}")

        for dc in self.dcs.values():
            # cập nhật util đến end_time
            if 0.0 < dc.util_last_ts < self.end_time:
                dt = self.end_time - dc.util_last_ts
                dc.util_gpu_time += dc.busy_gpus * dt
                dc.util_last_ts = self.end_time
            dc.accrue_energy(self.end_time)

        if self._pbar is not None:
            if self._pbar.n < self._pbar.total:
                self._pbar.update(self._pbar.total - self._pbar.n)
            self._pbar.close()

    def _net_tuple(self, src: str, dst: str, job) -> tuple[float, list[str], float, float, float]:
        """
        Trả về (Lnet_s, path, bottleneck_Gbps, path_cost_per_GB, transfer_s).
        transfer_s = Lnet_s + (payload_GB / bottleneck_Gbps) nếu có bottleneck hữu hạn.
        """
        Lnet_s, path, bottleneck, cost_sum = self.graph.shortest_path_latency(src, dst)
        # payload ước lượng theo loại job
        data_gb = 0.05 if job.jtype == 'inference' else 5.0
        if bottleneck and bottleneck > 0.0:
            xfer_s = data_gb / bottleneck
        else:
            # Graph trả về 0.0 nếu bottleneck là vô cực -> coi như không tính phần băng thông
            xfer_s = 0.0
        transfer_s = Lnet_s + xfer_s
        return Lnet_s, path, bottleneck, cost_sum, transfer_s

    def _rl_reallocate_training_jobs(self, dc: DataCenter, preempted_jobs: List[PreemptedJob]):
        if self.algo not in ("rl_energy_upgr") or not self.rl_upgr:
            return

        for preempted_job in preempted_jobs:
            job = preempted_job.job

            if self.algo == "rl_energy_upgr" and (self.rl_upgr is not None):
                # RL Upgr chỉ quyết định số GPU; f lấy theo energy-opt như joint_nf
                obs = self._upgr_obs()
                m_dc, m_g = self._upgr_masks()
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                m_dc_t = torch.from_numpy(m_dc).unsqueeze(0).bool()
                m_g_t = torch.from_numpy(m_g).unsqueeze(0).bool()
                a = self.rl_upgr.select_action(obs_t, m_dc_t, m_g_t, deterministic=False)

                # n từ actor (kẹp theo free/limit)
                n_rl = self._gidx_to_n(int(a["g"]))
                n_rl = max(1, min(n_rl, dc.free_gpus, self.policy.max_gpus_per_job))

                # f tối ưu năng lượng tại n_rl (giống joint_nf) + guard deadline nếu có
                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                levels = sorted(dc.freq_levels) if dc.freq_levels else [getattr(dc, "current_freq", 1.0)]
                f_rl = best_energy_freq(n_rl, levels, pC, tC)

                ddl = getattr(job, "deadline", None)  # training thường None; nếu có thì guard
                if ddl is not None:
                    if job.size * step_time_s(n_rl, f_rl, tC) > ddl:
                        for f_cand in levels:
                            if job.size * step_time_s(n_rl, f_cand, tC) <= ddl:
                                f_rl = f_cand
                                break
                        else:
                            f_rl = levels[-1]

            # Thực sự resume 1 lần tại đây cho mọi nhánh
            self._resume_preempted_job(dc, preempted_job, n_rl, f_rl)

    # --- arrivals at ingress ---
    def _handle_ingress_arrival(self, jtype: str, ing_name: str):
        # ing = self.ingresses[ing_name]
        jid = next(self.jid_counter)
        size = sample_job_size(jtype)
        job = Job(jid=jid, ingress=ing_name, jtype=jtype, size=size, arrival_time=self.now, dc_name=None)

        # ----- ROUTING: chọn DC theo algo -----
        if self.algo == "eco_route":
            best = None
            for dc in self.dcs.values():
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc.name, job)
                score, n_star, f_star = self._score_dc_for_job(dc, job)  # đã có trước
                cand = (score, dc.name, Lnet, path, bottleneck, cost_sum, transfer_s, n_star, f_star)
                if (best is None) or (cand[0] < best[0]):
                    best = cand
            _, dc_name, Lnet, path, bottleneck, cost_sum, transfer_s, n_star, f_star = best
            job._eco_hint = (n_star, f_star)

        elif self.algo == "rl_energy_upgr" and (self.rl_upgr is not None):
            # Dùng RL Upgr để chọn DC + nGPU + freq
            obs = self._upgr_obs()
            m_dc, m_g = self._upgr_masks()

            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            m_dc_t = torch.from_numpy(m_dc).unsqueeze(0).bool()
            m_g_t = torch.from_numpy(m_g).unsqueeze(0).bool()

            a = self.rl_upgr.select_action(obs_t, m_dc_t, m_g_t, deterministic=False)
            dc_name = self._dc_idx_to_name(a["dc"])
            n_sel = self._gidx_to_n(a["g"])
            # f_sel = float(a["f"])

            # lưu dấu vết để train khi job xong
            job._upgr_state0 = obs
            job._upgr_action = {"dc_idx": int(a["dc"]), "g_idx": int(a["g"]), "n": n_sel}

            Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)
        else:
            names = list(self.dcs.keys())
            dc_name = random.choice(names)
            Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)

        # ----- LỊCH SỰ KIỆN chuyển tới DC -----
        self._schedule(self.now + transfer_s, 'xfer_done', {
            'ing': ing_name,
            'dc': dc_name,
            'jid': jid,
            'job': job,
            'net_lat_s': Lnet,
            'net_bw_Gbps': bottleneck,
            'net_path_cost_GB': cost_sum
        })

        # ----- LỊCH ARRIVAL TIẾP THEO ở ingress này -----
        ia = (self.arr_inf if jtype == 'inference' else self.arr_trn).next_interarrival(self.now)
        self._schedule(self.now + ia, 'arrival_inf' if jtype == 'inference' else 'arrival_trn', {'ing': ing_name})

    # --- after WAN transfer ---
    def _handle_transfer_done(self, payload: dict):
        dc = self.dcs[payload['dc']]
        job: Job = payload['job']
        job.dc_name = dc.name
        job.arrival_time = self.now  # coi như 'đến' DC lúc này
        job.net_latency_s = payload['net_lat_s']
        # enqueue or start
        if dc.free_gpus > 0:
            if self.algo == "joint_nf":
                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                # deadline cho inference, còn training thì None
                ddl = getattr(job, "deadline", None)
                # mục tiêu "energy" ở đây (hoặc "carbon" nếu muốn carbon-aware)
                n_star, f_star, *_ = best_nf_grid(
                    n_max=self.policy.max_gpus_per_job,
                    freq_levels=dc.freq_levels,
                    p_coeffs=pC, t_coeffs=tC,
                    objective="energy",
                    carbon_intensity=0.0,
                    deadline_s=ddl
                )
                return self._start_job_with_nf(dc, job, n_star, f_star)
            elif self.algo == "bandit":
                # chọn n theo policy (vd ưu tiên 1 cho inference, 2-4 cho training)
                n_try = min(dc.free_gpus, self.policy.max_gpus_per_job)
                f_try = self.bandit.select(dc.name, job.jtype, dc.freq_levels)
                return self._start_job_with_nf(dc, job, n_try, f_try)
            elif self.algo == "carbon_cost":
                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                price = self._price_kwh(dc.name)  # <— dùng helper
                CI = self.carbon.get(dc.name, 0.0)
                # Nếu có bảng giá -> tối ưu chi phí; nếu không -> tối ưu carbon
                if price > 0.0:
                    n_star, f_star, *_ = best_nf_grid(
                        n_max=self.policy.max_gpus_per_job,
                        freq_levels=dc.freq_levels,
                        p_coeffs=pC, t_coeffs=tC,
                        objective="cost",
                        price_kwh=price,
                        deadline_s=getattr(job, "deadline", None)
                    )
                else:
                    n_star, f_star, *_ = best_nf_grid(
                        n_max=self.policy.max_gpus_per_job,
                        freq_levels=dc.freq_levels,
                        p_coeffs=pC, t_coeffs=tC,
                        objective="carbon",
                        carbon_intensity=CI,
                        deadline_s=getattr(job, "deadline", None)
                    )
                return self._start_job_with_nf(dc, job, n_star, f_star)
            elif self.algo == "rl_energy_upgr" and hasattr(job, "_upgr_action"):
                a = job._upgr_action
                # đảm bảo n không vượt free và policy
                n = max(1, min(a["n"], dc.free_gpus, self.policy.max_gpus_per_job))

                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                levels = sorted(dc.freq_levels) if getattr(dc, "freq_levels", None) else [
                    getattr(dc, "current_freq", 1.0)]
                f_opt = best_energy_freq(n, levels, pC, tC)

                ddl = getattr(job, "deadline", None)
                if ddl is not None:
                    # nếu f_opt khiến thời gian vượt deadline → nâng f lên mức nhỏ nhất thỏa deadline
                    if job.size * step_time_s(n, f_opt, tC) > ddl:
                        for f_cand in levels:
                            if job.size * step_time_s(n, f_cand, tC) <= ddl:
                                f_opt = f_cand
                                break
                        else:
                            f_opt = levels[-1]  # vẫn không đủ → dùng f lớn nhất

                return self._start_job_with_nf(dc, job, n, float(f_opt))
            elif self.algo == "debug":
                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                n = self.num_fixed_gpus
                f = self.fixed_freq if self.fixed_freq else best_energy_freq(n, dc.freq_levels, pC, tC)
                return self._start_job_with_nf(dc, job, n, f)
            else:
                gpus = select_gpus_and_set_freq(dc, job, self.policy)
                if gpus > 0:
                    return self._start_job(dc, job, gpus)

        (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job)

    def _start_job(self, dc: DataCenter, job: Job, gpus: int):
        if gpus <= 0:
            (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job)
            return
        dc.busy_gpus += gpus
        dc.running_jobs[job.jid] = (job, gpus)
        job.gpus_assigned = gpus
        job.start_time = self.now

        job.f_used = dc.current_freq
        job.units_total = job.size
        job.units_done = 0.0
        job.last_update = self.now
        job.ev_gen += 1

        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        T_unit = step_time_s(gpus, dc.current_freq, t_coeffs)

        # service_time = job.size * T_unit
        self._schedule(self.now + job.size * T_unit, 'job_finish', {'dc': dc.name, 'jid': job.jid, 'gen': job.ev_gen})

    def _handle_job_finish(self, dc_name: str, jid: int):
        dc = self.dcs[dc_name]
        tup = dc.running_jobs.pop(jid, None)
        if not tup:
            return
        job, g = tup
        dc.busy_gpus = max(0, dc.busy_gpus - g)
        job.finish_time = self.now

        # Add remaining unit of job to accumulate upon job finish
        self._log_accumulate_job_unit(dc, job, job.finish_time % self.log_interval)

        # Dùng f_used nếu đã bật per-job DVFS; fallback = tần số DC hiện tại
        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        f_used = getattr(job, "f_used", dc.current_freq)
        T_pred, P_pred, E_pred = energy_tuple(g, f_used, p_coeffs, t_coeffs)  # per-unit

        # ====== Chuẩn bị metrics cho RL (energy/latency/tail/power/...)
        # Giả định E_pred là năng lượng (Joule) mỗi "unit" → đổi sang kWh
        J_TO_KWH = 1.0 / 3.6e6
        units_done = float(job.size)
        E_job_kwh = float(E_pred) * units_done * J_TO_KWH

        # Sojourn time của job (s) → ms
        sojourn_s = max(0.0, job.finish_time - job.start_time)
        sojourn_ms = 1000.0 * sojourn_s

        # Theo dõi tail latency (P99) bằng cửa sổ trượt
        if not hasattr(self, "_lat_hist"):
            self._lat_hist = {"inference": [], "training": []}
        buf = self._lat_hist.setdefault(job.jtype, [])
        buf.append(sojourn_s)
        # Giữ ~2k mẫu gần nhất
        if len(buf) > 2048:
            del buf[: len(buf) - 2048]
        mean_ms = (sum(buf) / len(buf)) * 1000.0 if buf else sojourn_ms
        p99_ms = float(np.percentile(buf, 99) * 1000.0) if len(buf) >= 5 else sojourn_ms

        # Công suất hiện tại (xấp xỉ trung bình trong step) và số lần chuyển trạng thái nguồn
        if self.power_cap > 0:
            P_now = self._estimate_dc_power(dc, getattr(dc, "current_freq", 1.0))
        else:
            P_now = self._estimate_dc_power(dc, getattr(dc, "current_freq", 1.0))
        power_state_changes = int(getattr(self, "_power_state_changes_step", 0))

        # Metrics đưa cho agent (agent sẽ tự tính reward theo mode)
        rl_metrics = {
            "energy_kwh": E_job_kwh,
            "units_processed": units_done,
            "mean_latency_ms": mean_ms,
            "p99_latency_ms": p99_ms,
            "power_W": float(P_now),
            "power_state_changes": power_state_changes,
        }
        self.logger.debug({k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in rl_metrics.items()})

        # === Update RL Upgr ===
        if (self.algo == "rl_energy_upgr" and self.rl_upgr is not None and
                hasattr(job, "_upgr_action") and hasattr(job, "_upgr_state0")):
            # TODO Mode 1
            # r = - (float(rl_metrics["energy_kwh"]) / (float(rl_metrics["units_processed"] + 1e-9)))

            # TODO Mode 2
            E_unit_kWh = float(rl_metrics["energy_kwh"]) / (float(rl_metrics["units_processed"]) + 1e-9)

            # --- ưu tiên ít GPU và không vượt f_opt năng lượng ---
            n = max(1, int(job._upgr_action["n"]))

            # 1/n: càng ít GPU càng thưởng nhẹ (scale nhỏ để không phá SLA)
            # TODO: Chỉnh alpha
            alpha_n = 0.05
            gpu_pref = alpha_n * (1.0 / n)

            r = - E_unit_kWh + gpu_pref

            sla_ms_target = self.rl_upgr.cmdp.constraints["latency_p99"].target
            n_min = self._min_n_for_sla(dc, job, f_used, sla_ms_target)
            gpu_over = max(0, g - n_min)

            # Costs cho CMDP
            costs = {
                "latency_p99": float(rl_metrics["p99_latency_ms"]),
                "power": float(rl_metrics["power_W"]),
                "gpu_over": float(gpu_over),
            }

            s0 = job._upgr_state0
            s1 = self._upgr_obs()  # trạng thái sau khi job hoàn tất (next state)
            a_dc_idx = int(job._upgr_action["dc_idx"])
            a_g_idx = int(job._upgr_action["g_idx"])

            # mask tại thời điểm chọn action (để agent có thể dùng trong training nếu cần)
            m_dc, m_g = self._upgr_masks()

            self._replay.add(Transition(
                s=s0, s_next=s1,
                a_dc=a_dc_idx, a_g=a_g_idx,
                r=r, costs=costs, done=True,
                mask_dc=m_dc, mask_g=m_g
            ))

            # Train khi đủ dữ liệu
            if self._replay.size >= self._upgr_warmup:
                batch = self._replay.sample(self._upgr_batch)
                stats = self.rl_upgr.train_step(batch)
                if stats:
                    self.logger.info({k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in stats.items()})

            # dọn dấu vết
            del job._upgr_state0
            del job._upgr_action

        # Ghi log job (giữ nguyên)
        with open(self.job_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                job.jid, getattr(job, "ingress", ""),
                job.jtype, f"{job.size:.4f}", dc.name,
                f"{f_used:.3f}", g, f"{getattr(job, 'net_latency_s', 0.0):.4f}",
                f"{job.start_time:.6f}", f"{job.finish_time:.6f}",
                f"{(job.finish_time - job.start_time):.6f}",
                f"{job.preempt_count}",
                f"{T_pred:.6f}", f"{P_pred:.2f}", f"{E_pred:.2f}"
            ])

        # Bandit: cập nhật reward = - energy_per_unit (hoặc đổi sang carbon/cost nếu muốn)
        if getattr(self, "bandit", None) is not None:
            self.bandit.update(dc.name, job.jtype, f_used, E_pred)

        # Elastic scaling - after a job completion (giữ nguyên)
        preempted_jobs = None
        if self.elastic_scaling:
            should_reallocate = self._should_reallocation(dc, job.jtype)
            if should_reallocate and job.jtype == "training":
                preempted_jobs = self._preempt_all_training_jobs(dc, f"Re-allocate on job completion.")
                self.logger.info(f"Preempt all training jobs of {dc.name} at {self._current_hour()} upon job completion.")
            if preempted_jobs:
                self._rl_reallocate_training_jobs(dc, preempted_jobs)

        # start next jobs (inference first) (giữ nguyên)
        while dc.free_gpus > 0:
            nxt = None
            if self.policy.inf_priority and dc.q_inf:
                nxt = dc.q_inf.pop(0)
            elif dc.q_train:
                nxt = dc.q_train.pop(0)
            if nxt is None:
                break

            if self.algo == "rl_energy_upgr" and (self.rl_upgr is not None):
                obs = self._upgr_obs()
                m_dc, m_g = self._upgr_masks()
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                m_dc_t = torch.from_numpy(m_dc).unsqueeze(0).bool()
                m_g_t = torch.from_numpy(m_g).unsqueeze(0).bool()
                a = self.rl_upgr.select_action(obs_t, m_dc_t, m_g_t, deterministic=False)

                dc_tgt = self.dcs[self._dc_idx_to_name(a["dc"])]
                if dc_tgt.free_gpus <= 0:
                    # không DC nào rảnh phù hợp → đẩy lại hàng đợi và thoát vòng
                    (dc.q_inf if nxt.jtype == 'inference' else dc.q_train).insert(0, nxt)
                    break

                n_sel = max(1, min(self._gidx_to_n(int(a["g"])), dc_tgt.free_gpus, self.policy.max_gpus_per_job))

                # === Chọn f tối ưu về năng lượng như joint_nf, với guard SLA nếu có ===
                pC, tC = self.coeffs_map[(dc_tgt.name, nxt.jtype)]
                levels = sorted(dc_tgt.freq_levels)

                # f tối ưu năng lượng tại n_sel
                f_opt = best_energy_freq(n_sel, levels, pC, tC)

                # Guard theo deadline (nếu job có)
                ddl = getattr(nxt, "deadline", None)  # deadline (s) nếu có
                if ddl is not None:
                    # nếu f_opt khiến thời gian vượt deadline → tăng f lên mức nhỏ nhất thoả
                    if nxt.size * step_time_s(n_sel, f_opt, tC) > ddl:
                        for f_cand in levels:
                            if nxt.size * step_time_s(n_sel, f_cand, tC) <= ddl:
                                f_opt = f_cand
                                break
                        else:
                            f_opt = levels[-1]  # vẫn không đủ thì dùng f lớn nhất

                f_sel = float(f_opt)

                # start và ghi dấu vết để update khi xong job
                self._start_job_with_nf(dc_tgt, nxt, n_sel, f_sel)
                nxt._upgr_state0 = obs
                nxt._upgr_action = {"dc_idx": int(a["dc"]), "g_idx": int(a["g"]), "n": n_sel}
                break  # DC có thể đã thay đổi → thoát để tránh giả định free_gpus cũ

            if self.algo == "joint_nf":
                pC, tC = self.coeffs_map[(dc.name, nxt.jtype)]
                n_star, f_star, *_ = best_nf_grid(
                    n_max=self.policy.max_gpus_per_job,
                    freq_levels=dc.freq_levels,
                    p_coeffs=pC, t_coeffs=tC,
                    objective="energy",
                    carbon_intensity=0.0,
                    deadline_s=getattr(nxt, "deadline", None)
                )
                self._start_job_with_nf(dc, nxt, n_star, f_star)

            elif self.algo == "bandit":
                n_try = min(dc.free_gpus, self.policy.max_gpus_per_job)
                f_try = self.bandit.select(dc.name, nxt.jtype, dc.freq_levels)
                self._start_job_with_nf(dc, nxt, n_try, f_try)

            elif self.algo == "carbon_cost":
                pC, tC = self.coeffs_map[(dc.name, nxt.jtype)]
                CI = self.carbon.get(dc.name, 0.0)
                n_star, f_star, *_ = best_nf_grid(
                    n_max=self.policy.max_gpus_per_job,
                    freq_levels=dc.freq_levels,
                    p_coeffs=pC, t_coeffs=tC,
                    objective="carbon",
                    carbon_intensity=CI,
                    deadline_s=getattr(nxt, "deadline", None)
                )
                self._start_job_with_nf(dc, nxt, n_star, f_star)

            else:
                gpus = select_gpus_and_set_freq(dc, nxt, self.policy)
                if gpus <= 0:
                    (dc.q_inf if nxt.jtype == 'inference' else dc.q_train).insert(0, nxt)
                    break
                self._start_job(dc, nxt, gpus)

    def _handle_log(self, interval: float):
        with open(self.cluster_log_path, 'a', newline='') as f:
            w = csv.writer(f)
            for name, dc in self.dcs.items():
                run_total = len(dc.running_jobs)
                run_inf = sum(1 for j, _ in dc.running_jobs.values() if j.jtype == 'inference')
                run_trn = run_total - run_inf
                util_inst = (dc.busy_gpus / dc.total_gpus) if dc.total_gpus else 0.0
                elapsed = max(1e-9, self.now - (dc.util_begin_ts or self.now))
                util_avg = (dc.util_gpu_time / (dc.total_gpus * elapsed)) if dc.total_gpus else 0.0
                power_now = self._estimate_dc_power(dc, getattr(dc, "current_freq", 1.0))
                # Accumulate job_size of all running_jobs upon log event
                for job, _ in dc.running_jobs.values():
                    self._log_accumulate_job_unit(dc, job, interval)

                w.writerow([f"{self.now:.3f}", name, f"{dc.current_freq:.2f}",
                            dc.busy_gpus, dc.free_gpus, run_total, run_inf, run_trn,
                            len(dc.q_inf), len(dc.q_train),
                            f"{util_inst:.4f}", f"{util_avg:.4f}", f"{dc.accumulated_job_unit:.4f}",
                            f"{power_now:.2f}", f"{dc.energy_joules / 1000.0:.4f}"])
        self._schedule(self.now + interval, 'log', {'interval': interval})

    def _log_accumulate_job_unit(self, dc: DataCenter, job: Job, accumulate_time: float):
        """Accumulate job_size = 1 / T(n,f) x time"""
        _, tC = self.coeffs_map[(dc.name, job.jtype)]
        n = job.gpus_assigned
        f = job.f_used
        tpt_job = 1 / step_time_s(n, f, tC)

        dc.accumulated_job_unit += (tpt_job * accumulate_time)

    def _start_job_with_nf(self, dc, job, n, f):
        # cấp phát
        n = max(1, min(n, dc.free_gpus))
        if n <= 0:
            (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job)
            return
        dc.busy_gpus += n
        dc.running_jobs[job.jid] = (job, n)
        job.gpus_assigned = n
        job.start_time = self.now
        # per-job DVFS (nếu đã thêm f_used/ev_gen, dùng như sau; nếu chưa, coi như DC-level)
        if hasattr(job, "f_used"):
            job.f_used = f
            job.units_total = job.size
            job.units_done = 0.0
            job.last_update = self.now
            job.ev_gen += 1
        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        T_unit = step_time_s(n, f, t_coeffs)
        self._schedule(self.now + job.size * T_unit, 'job_finish',
                       {'dc': dc.name, 'jid': job.jid, 'gen': getattr(job, "ev_gen", 0)})

    def _current_hour(self) -> int:
        """Giờ hiện tại (0..23) trong ngày mô phỏng, tính theo self.now (giây)."""
        return int((self.now % 86400) // 3600)

    def _price_kwh(self, dc_name: str) -> float:
        """
        Giá điện (USD/kWh) tại giờ hiện tại cho DC.
        Hỗ trợ 2 format:
          - Global hourly map: {hour:int -> price:float}
          - Per-DC hourly map: {dc_name:str -> {hour:int -> price:float}}
        """
        ep = getattr(self, "energy_price", {}) or {}
        h = self._current_hour()

        # Global hourly map? (keys là int)
        if ep and all(isinstance(k, int) for k in ep.keys()):
            return float(ep.get(h, 0.0))

        # Per-DC hourly map? (keys là tên DC)
        dc_map = ep.get(dc_name)
        if isinstance(dc_map, dict):
            return float(dc_map.get(h, 0.0))

        return 0.0

    def _score_dc_for_job(self, dc, job) -> tuple:
        """
        Trả về (score, n_star, f_star) cho job nếu chạy ở dc theo eco-objective.
        score đơn vị:
          - energy: Joule/job
          - carbon: gCO2/job
          - cost: USD/job
        """
        pC, tC = self.coeffs_map[(dc.name, job.jtype)]
        obj = getattr(self, "eco_objective", "energy")
        ddl = getattr(job, "deadline", None)

        # Carbon / price lấy từ state đã có
        CI = self.carbon.get(dc.name, 0.0)
        price = self._price_kwh(dc.name) if hasattr(self, "_price_kwh") else 0.0

        # chọn (n,f) tốt nhất cho mục tiêu
        if obj == "energy":
            n, f, T, P, E_unit = best_nf_grid(self.policy.max_gpus_per_job, dc.freq_levels, pC, tC,
                                              objective="energy", deadline_s=ddl)
            score = E_unit * job.size  # J / job

        elif obj == "carbon":
            n, f, T, P, E_unit = best_nf_grid(self.policy.max_gpus_per_job, dc.freq_levels, pC, tC,
                                              objective="carbon", carbon_intensity=CI, deadline_s=ddl)
            score = (E_unit * job.size) / 3.6e6 * CI  # kWh * gCO2/kWh = gCO2

        else:  # cost
            n, f, T, P, E_unit = best_nf_grid(self.policy.max_gpus_per_job, dc.freq_levels, pC, tC,
                                              objective="cost", price_kwh=price, deadline_s=ddl)
            score = (E_unit * job.size) / 3.6e6 * price  # USD

        return score, n, f

    def _upgr_obs(self) -> np.ndarray:
        """Vector hoá trạng thái cho RL Upgr: [now] + cho từng DC: [total, busy, free, current_f, qlen_inf, qlen_trn]."""
        feats = []
        for dc in self.dcs.values():
            total = float(getattr(dc, "total_gpus", dc.total_gpus))
            busy = float(dc.busy_gpus)
            free = max(0.0, total - busy)
            cf = float(getattr(dc, "current_freq", self._f_min))
            q_inf = float(len(dc.q_inf))
            q_trn = float(len(dc.q_train))
            feats.extend([total, busy, free, cf, q_inf, q_trn])
        now = float(self.now)
        return np.asarray([now] + feats, dtype=np.float32)

    def _upgr_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mask cho DC (có GPU rảnh) và cho #GPU (1..N) theo free tối đa trên mọi DC."""
        dc_mask = []
        max_free = 0
        for dc in self.dcs.values():
            total = int(getattr(dc, "total_gpus", dc.total_gpus))
            free = max(0, total - int(dc.busy_gpus))
            dc_mask.append(free > 0)
            max_free = max(max_free, free)

        n_choices = self._n_g_choices
        g_mask = [(i + 1) <= max_free for i in range(n_choices)]

        # --- chọn buffer phù hợp với workload hiện tại ---
        buf = None
        if hasattr(self, "_lat_hist"):
            # nếu có training thì ưu tiên training; nếu không có thì dùng inference
            buf = self._lat_hist.get("training") or self._lat_hist.get("inference")

        if buf and len(buf) >= 5 and hasattr(self, "rl_upgr"):
            p99_recent_ms = float(np.percentile(buf, 99) * 1000.0)
            target = self.rl_upgr.cmdp.constraints["latency_p99"].target
            # chỉ thu nhỏ #GPU khi hệ đang "dư" SLO rõ rệt
            if p99_recent_ms < 0.9 * target:
                cap = 1  # ưu tiên ít GPU
                g_mask = np.array([(i + 1) <= min(cap, max_free) for i in range(n_choices)], dtype=bool)

        return np.asarray(dc_mask, bool), np.asarray(g_mask, bool)

    def _dc_idx_to_name(self, idx: int) -> str:
        return self._dc_names[int(idx)]

    def _gidx_to_n(self, g_idx: int) -> int:
        """Map index 0..(N-1) -> số GPU 1..N"""
        return int(g_idx) + 1

    def _min_n_for_sla(self, dc: DataCenter, job: Job, f: float, sla_ms: float) -> int:
        _, tC = self.coeffs_map[(dc.name, job.jtype)]
        for n_try in range(1, self.policy.max_gpus_per_job + 1):
            if job.size * step_time_s(n_try, f, tC) * 1000.0 <= sla_ms:
                return n_try
        return self.policy.max_gpus_per_job
