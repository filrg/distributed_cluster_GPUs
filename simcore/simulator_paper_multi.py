import csv, heapq, itertools, random, os
from typing import Dict, List, Tuple, Optional, Union
from .models import Job, DataCenter, PreemptedJob
from .arrivals import ArrivalConfig
from .policy import PolicyConfig, select_gpus_and_set_freq
from .latency_paper import step_time_s
from .policy_paper import energy_tuple, best_nf_grid, best_energy_freq
from .network import Ingress, Graph
from .router import RouterPolicy, select_dc
from .energy_paper import task_power_w, gpu_power_w
from .learners import BanditDVFS
from .freq_load_agg import TaskState, aggregate_with_atoms
from .learners_rl import RLEnergyAgent
from .learners_rl_adv import RLEnergyAgentAdv, RLAction

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
                 control_interval: float = 5.0,
                 show_progress: bool = True,
                 rl_alpha=0.1, rl_gamma=0.0, rl_eps=0.2, rl_eps_decay=0.995, rl_eps_min=0.02, rl_n_cand=2,
                 rl_tau=0.1, rl_clip_grad=5.0, rl_baseline_beta=0.01):
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
        random.seed(rng_seed)
        self.jid_counter = itertools.count(1)

        self.cluster_log_path = "cluster_log.csv"
        self.job_log_path = "job_log.csv"
        if log_path:
            out_dir = os.path.join(log_path, algo)
            os.makedirs(out_dir, exist_ok=True)
            self.cluster_log_path = os.path.join(out_dir, "cluster_log.csv")
            self.job_log_path = os.path.join(out_dir, "job_log.csv")

        self.algo = algo
        self.elastic_scaling = elastic_scaling and self.algo in ("rl_energy", "rl_energy_adv")
        self.rl = None
        self.rl_n_cand = int(rl_n_cand)
        if self.algo == "rl_energy":
            self.rl = RLEnergyAgent(alpha=rl_alpha, gamma=rl_gamma, eps=rl_eps,
                                    eps_decay=rl_eps_decay, eps_min=rl_eps_min)
        elif self.algo == "rl_energy_adv":
            self.rl = RLEnergyAgentAdv(alpha=rl_alpha, gamma=rl_gamma,
                                       tau=getattr(self, "rl_tau", 0.1) if hasattr(self, "rl_tau") else 0.1,
                                       eps=rl_eps, eps_decay=rl_eps_decay, eps_min=rl_eps_min,
                                       clip_grad=getattr(self, "rl_clip_grad", 5.0) if hasattr(self, "rl_clip_grad") else 5.0,
                                       baseline_beta=getattr(self, "rl_baseline_beta", 0.01) if hasattr(self, "rl_baseline_beta") else 0.01)
        self.power_cap = power_cap
        self.control_interval = control_interval
        self.bandit = BanditDVFS(init_explore=1, objective="energy") if algo == "bandit" else None

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
                smoothing=0.3
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
        job.last_update = self.now

        preempt_duration = self.now - preempted.preempt_time
        job.total_preempt_time += preempt_duration
        # Resume
        dc.busy_gpus += n_resume #dc.busy_gpus += job.gpus_assigned
        dc.running_jobs[job.jid] = (job, n_resume)

        units_left = max(0.0, job.units_total - job.units_done)
        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        T_unit = step_time_s(job.gpus_assigned, job.f_used, t_coeffs)
        finish_time = units_left / max(1.0 / T_unit, 1e-9)

        job.ev_gen += 1
        self._schedule(self.now + finish_time, "job_finish",
                       {'dc': dc.name, 'jid': job.jid, 'gen': job.ev_gen})
        dc.preempted_jobs.remove(preempted)
        #print(f"Resume job {job.jid} at {self._current_hour()}.")

    def _should_reallocation(self, dc: DataCenter, job_type: str) -> bool:
        """Re-allocate on train completion (only with elastic_scaling)"""
        if job_type != "training":
            return False
        # if self.algo not in ("rl_energy", "rl_energy_adv"):
        #     return False

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

    # --- run loop ---
    def run(self):
        with open(self.cluster_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["time_s", "dc", "freq", "busy", "free",
                                    "run_total", "run_inf", "run_train",
                                    "q_inf", "q_train",
                                    "util_inst", "util_avg",
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
            # Graph của bạn trả 0.0 nếu bottleneck là vô cực -> coi như không tính phần băng thông
            xfer_s = 0.0
        transfer_s = Lnet_s + xfer_s
        return Lnet_s, path, bottleneck, cost_sum, transfer_s

    # TODO
    def _rl_build_state(self, dc, job, net_lat_est: float) -> dict:
        return {
            "job_type": job.jtype,
            "job_size": job.size,
            "dc_total": dc.total_gpus,
            "dc_busy": dc.busy_gpus,
            "dc_q_inf": len(dc.q_inf),
            "dc_q_trn": len(dc.q_train),
            "net_lat": net_lat_est,
            "f_levels_len": len(dc.freq_levels)
        }

    def _rl_actions_for_job(self, job) -> list[RLAction]:
        acts = []
        for dc in self.dcs.values():
            if not dc.freq_levels:
                continue
            # lấy vài mức f tiêu biểu: min, median, max
            f_levels = sorted(dc.freq_levels)
            candidates_f = {f_levels[0], f_levels[-1], f_levels[len(f_levels) // 2]}
            # ứng viên n: 1..rl_n_cand, cắt bởi free_gpus và policy.max_gpus_per_job
            max_n = min(self.policy.max_gpus_per_job, dc.total_gpus)  # dùng total; lúc route chưa chắc free
            for n in range(1, min(self.rl_n_cand, max_n) + 1):
                for f in candidates_f:
                    ddl = getattr(job, "deadline", None)
                    if ddl is not None:
                        tC = self.coeffs_map[(dc.name, job.jtype)][1]
                        T_unit = step_time_s(n, f, tC)
                        if T_unit > ddl:
                            continue
                    acts.append(RLAction(dc.name, n, f))
        return acts

    def _rl_reallocate_training_jobs(self, dc: DataCenter, preempted_jobs: List[PreemptedJob]):
        if self.algo not in ("rl_energy", "rl_energy_adv") or not self.rl:
            return

        for preempted_job in preempted_jobs:
            job = preempted_job.job
            
            if self.algo == "rl_energy":
                state_map = {
                    dc.name: self._rl_build_state(dc, job, 0.0)
                }
                actions = self._rl_actions_for_job(job)
                if actions:
                    a = self.rl.select_per_dc(state_map, actions)
                    job._rl_state0 = state_map[a.dc_name]
                    job._rl_action = a
                    n_rl = a.n
                    f_rl = a.f
                else:
                    # Fallback to best energy frequency
                    pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                    n_rl = 1  # Default to 1 GPU
                    f_rl = best_energy_freq(n_rl, dc.freq_levels, pC, tC)
                    
            elif self.algo == "rl_energy_adv":
                base_state = self._rl_build_state(dc, job, 0.0)
                base_state["price_kwh"] = self._price_kwh(dc.name)
                base_state["carbon_g_per_kwh"] = self.carbon.get(dc.name, 0.0)
                cap_headroom = max(0.0, self.power_cap - self._estimate_dc_power(dc, getattr(dc, "current_freq", 1.0))) if self.power_cap > 0 else 0.0
                base_state["cap_headroom_W"] = cap_headroom
                
                actions = self._rl_actions_for_job(job)
                if actions:
                    # Augment state for each action
                    state_map = {}
                    for a in actions:
                        s = dict(base_state)
                        pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                        T_unit = step_time_s(a.n, a.f, tC)
                        P_est = a.n * gpu_power_w(a.f, pC)
                        s["t_unit"] = T_unit
                        s["p_est"] = P_est
                        s["e_unit"] = P_est * T_unit
                        state_map[dc.name] = s
                    
                    a = self.rl.select_per_dc(state_map, actions)
                    job._rl_state0 = state_map[dc.name]
                    job._rl_action = a
                    n_rl = a.n
                    f_rl = a.f
                else:
                    # Fallback to best energy frequency
                    pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                    n_rl = 1
                    f_rl = best_energy_freq(n_rl, dc.freq_levels, pC, tC)

            #self._start_job_with_nf(dc, job, n_rl, f_rl)
            self._resume_preempted_job(dc, preempted_job, n_rl, f_rl)

    # --- arrivals at ingress ---
    def _handle_ingress_arrival(self, jtype: str, ing_name: str):
        ing = self.ingresses[ing_name]
        jid = next(self.jid_counter)
        size = self._sample_job_size(jtype)
        job = Job(jid=jid, jtype=jtype, size=size, arrival_time=self.now, dc_name=None)

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

        elif self.algo == "rl_energy" and getattr(self, "rl", None) is not None:
            # build state per-DC + liệt kê action {(dc,n,f)}, chọn epsilon-greedy
            state_map = {}
            for dc in self.dcs.values():
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc.name, job)
                state_map[dc.name] = self._rl_build_state(dc, job, Lnet)

            actions = self._rl_actions_for_job(job)
            if actions:
                a = self.rl.select_per_dc(state_map, actions)
                job._rl_state0 = state_map[a.dc_name]
                job._rl_action = a
                dc_name = a.dc_name
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)
            else:
                # fallback: router cũ
                dc_name = list(self.dcs.keys())[0]
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)
                data_gb = 0.05 if job.jtype == 'inference' else 5.0
                xfer_s = (data_gb / bottleneck) if bottleneck > 0.0 else 0.0
                transfer_s = Lnet + xfer_s
                dc_name = list(self.dcs.keys())[0]
        elif self.algo == "rl_energy_adv" and getattr(self, "rl", None) is not None:
            # Build state per-DC (base) + augment per-action (t_unit/p_est/e_unit, price/carbon/cap_headroom)
            base_map = {}
            for dc in self.dcs.values():
                Lnet, path, bottleneck, cost_sum, transfer_s_tmp = self._net_tuple(ing_name, dc.name, job)
                base = self._rl_build_state(dc, job, Lnet)
                base["price_kwh"] = self._price_kwh(dc.name)
                base["carbon_g_per_kwh"] = self.carbon.get(dc.name, 0.0)
                cap_headroom = max(0.0, self.power_cap - self._estimate_dc_power(dc, getattr(dc, "current_freq",
                                                                                             1.0))) if self.power_cap > 0 else 0.0
                base["cap_headroom_W"] = cap_headroom
                base_map[dc.name] = base

            actions = self._rl_actions_for_job(job)
            if actions:
                # augment state theo từng action
                state_map = {}
                for a in actions:
                    s = dict(base_map[a.dc_name])
                    pC, tC = self.coeffs_map[(a.dc_name, job.jtype)]
                    T_unit = step_time_s(a.n, a.f, tC)
                    P_est = a.n * gpu_power_w(a.f, pC)
                    s["t_unit"] = T_unit
                    s["p_est"] = P_est
                    s["e_unit"] = P_est * T_unit
                    state_map[a.dc_name] = s
                a = self.rl.select_per_dc(state_map, actions)
                job._rl_state0 = state_map[a.dc_name]
                job._rl_action = a
                dc_name = a.dc_name
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)
            else:
                dc_name = list(self.dcs.keys())[0]
                Lnet, path, bottleneck, cost_sum, transfer_s = self._net_tuple(ing_name, dc_name, job)
        else:
            # baseline / cap_* / joint_nf / bandit / carbon_cost: dùng router cũ
            # Lưu ý: chọn DC xong vẫn phải lấy path/latency qua Graph.shortest_path_latency
            dc_name, Lnet, path, _score = select_dc(ing, jtype, self.dcs, self.coeffs_map,
                                                    self.graph, self.carbon, self.router_policy)
            # Nếu select_dc CHƯA trả Lnet theo Graph mới, thì cập nhật lại bằng Graph hiện tại:
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

    def _sample_job_size(self, jtype: str) -> float:
        import math, random
        if jtype == 'inference':
            xm, alpha = 1, 1.8 # 0.02, 2.5 - alpha < 2 -> infinite variance?
            u = max(1e-9, 1 - random.random())
            return xm / (u ** (1 / alpha))
        mu, sigma = math.log(20000), 0.4 # math.log(3.0), 0.6
        return max(0.1, random.lognormvariate(mu, sigma))

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
            elif self.algo == "rl_energy" and hasattr(job, "_rl_action"):
                a: RLAction = job._rl_action
                # khớp với thực tế free_gpus
                n = max(1, min(a.n, dc.free_gpus, self.policy.max_gpus_per_job))
                f = a.f
                return self._start_job_with_nf(dc, job, n, f)
            elif self.algo == "rl_energy_adv" and hasattr(job, "_rl_action"):
                a: RLAction = job._rl_action
                n = max(1, min(a.n, dc.free_gpus, self.policy.max_gpus_per_job))
                f = a.f
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
        # self._schedule(self.now + service_time, 'job_finish', {'dc': dc.name, 'jid': job.jid})
        self._schedule(self.now + job.size * T_unit, 'job_finish', {'dc': dc.name, 'jid': job.jid, 'gen': job.ev_gen})

    def _handle_job_finish(self, dc_name: str, jid: int):
        dc = self.dcs[dc_name]
        tup = dc.running_jobs.pop(jid, None)
        if not tup: return
        job, g = tup
        dc.busy_gpus = max(0, dc.busy_gpus - g)
        job.finish_time = self.now

        # Dùng f_used nếu đã bật per-job DVFS; fallback = tần số DC hiện tại
        p_coeffs, t_coeffs = self.coeffs_map[(dc.name, job.jtype)]
        f_used = getattr(job, "f_used", dc.current_freq)
        T_pred, P_pred, E_pred = energy_tuple(g, f_used, p_coeffs, t_coeffs)  # per-unit

        if self.algo in ("rl_energy", "rl_energy_adv") and (self.rl is not None) \
                and hasattr(job, "_rl_action") and hasattr(job, "_rl_state0"):
            E_job = E_pred * job.size
            # reward = -E_job (- 1000*laten) - 50*wait (- 0.1*overflow)
            reward = -float(E_job)
            ddl = getattr(job, "deadline", None)
            if ddl is not None:
                laten = max(0.0, (job.finish_time - job.start_time) - ddl)
                reward -= 1000.0 * laten
            wait = max(0.0, job.start_time - job.arrival_time)
            reward -= 50.0 * wait
            if self.power_cap > 0:
                P_now = self._estimate_dc_power(self.dcs[dc_name], getattr(self.dcs[dc_name], "current_freq", 1.0))
                overflow = max(0.0, P_now - self.power_cap)
                reward -= 0.1 * overflow
            next_state = self._rl_build_state(dc, job, 0.0)
            next_actions = self._rl_actions_for_job(job)
            self.rl.update(job._rl_state0, job._rl_action, reward, next_state, next_actions, done=True)

        with open(self.job_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                job.jid, getattr(job, "ingress", ""),
                job.jtype, f"{job.size:.4f}", dc.name,
                f"{f_used:.3f}", g, f"{getattr(job, 'net_latency_s', 0.0):.4f}",
                f"{job.start_time:.6f}", f"{job.finish_time:.6f}",
                f"{(job.finish_time - job.start_time):.6f}",
                f"{job.preempt_count:.6f}",
                f"{T_pred:.6f}", f"{P_pred:.2f}", f"{E_pred:.2f}"
            ])

        # Bandit: cập nhật reward = - energy_per_unit (hoặc đổi sang carbon/cost nếu muốn)
        if getattr(self, "bandit", None) is not None:
            self.bandit.update(dc.name, job.jtype, f_used, E_pred)

        # Elastic scaling - after a job completion
        preempted_jobs = None
        if self.elastic_scaling:
            should_reallocate = self._should_reallocation(dc, job.jtype)
            if should_reallocate and job.jtype == "training":
                preempted_jobs = self._preempt_all_training_jobs(dc, f"Re-allocate on job completion.")
            if preempted_jobs:
                self._rl_reallocate_training_jobs(dc, preempted_jobs)

        # start next jobs (inference first)
        while dc.free_gpus > 0:
            nxt = None
            if self.policy.inf_priority and dc.q_inf:
                nxt = dc.q_inf.pop(0)
            elif dc.q_train:
                nxt = dc.q_train.pop(0)
            if nxt is None: break

            # Nhánh theo thuật toán
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
                # baseline / cap_* : dùng policy cũ
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

                w.writerow([f"{self.now:.3f}", name, f"{dc.current_freq:.2f}",
                            dc.busy_gpus, dc.free_gpus, run_total, run_inf, run_trn,
                            len(dc.q_inf), len(dc.q_train),
                            f"{util_inst:.4f}", f"{util_avg:.4f}",
                            f"{power_now:.2f}", f"{dc.energy_joules / 1000.0:.4f}"])
        self._schedule(self.now + interval, 'log', {'interval': interval})

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
