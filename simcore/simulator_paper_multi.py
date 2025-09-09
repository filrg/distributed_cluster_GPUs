import csv, heapq, itertools, random
from typing import Dict, List, Tuple, Optional, Union
from .models import Job, DataCenter
from .arrivals import ArrivalConfig
from .policy import PolicyConfig, select_gpus_and_set_freq
from .coeffs import TrainPowerCoeffs, TrainLatencyCoeffs
from .latency_paper import step_time_s
from .policy_paper import energy_tuple, best_nf_grid
from .network import Ingress, Graph
from .router import RouterPolicy, select_dc
from .energy_paper import gpu_power_w, task_power_w
from .learners import BanditDVFS
from .freq_load_agg import TaskState, aggregate_with_atoms
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
                 rng_seed: int = 42,
                 algo: str = "baseline",
                 power_cap: float = 0.0,
                 control_interval: float = 5.0,
                 show_progress: bool = True):
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

        self.algo = algo
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
                mininterval=0.2,   # hạn chế spam stdout
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
        if self.algo not in ("cap_uniform", "cap_greedy"):
            return
        if self.power_cap <= 0:
            return

        # Ước lượng tổng P hiện tại bằng mô hình paper (đã có _estimate_dc_power)
        totalP = sum(self._estimate_dc_power(dc, dc.current_freq) for dc in self.dcs.values())
        if totalP <= self.power_cap:
            return

        deficit = totalP - self.power_cap

        if self.algo == "cap_uniform":
            # giữ nguyên như version cũ: hạ đồng loạt theo DC, mỗi bước chọn ∆P lớn nhất
            return self._cap_uniform(deficit)

        # ---- cap_greedy: chọn 'atoms' per-job theo aggregate ----
        tasks = []
        for dc in self.dcs.values():
            levels = dc.freq_levels
            for job, g in dc.running_jobs.values():
                pC, tC = self.coeffs_map[(dc.name, job.jtype)]
                cur_f = getattr(job, "f_used", 0.0) or dc.current_freq
                tasks.append(TaskState(job_id=job.jid, dc_name=dc.name, n=g, f=cur_f,
                                       freq_levels=levels, p_coeffs=pC, t_coeffs=tC))
        _, down_atoms = aggregate_with_atoms(tasks)

        for atom in down_atoms:
            if deficit <= 1e-6:
                break
            dc = self.dcs[atom.dc_name]
            tup = dc.running_jobs.get(atom.job_id)
            if not tup:
                continue
            job, g = tup
            # ÁP DỤNG BƯỚC: reschedule job với f_to
            self._reschedule_job(dc, job, g, atom.f_to)
            deficit -= atom.dP  # xấp xỉ; nếu muốn chính xác, ước lượng lại ∆P sau khi đổi f

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
            csv.writer(f).writerow(["jid","ingress","type","size","dc","f_used","n_gpus",
                                    "net_lat_s","start_s","finish_s","latency_s","T_pred","P_pred","E_pred"])

        while self.event_q:
            ev = self._pop()
            if ev is None: break
            t,_,etype,payload = ev
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

    # --- arrivals at ingress ---
    def _handle_ingress_arrival(self, jtype: str, ing_name: str):
        ing = self.ingresses[ing_name]
        jid = next(self.jid_counter)
        size = self._sample_job_size(jtype)
        job = Job(jid=jid, jtype=jtype, size=size, arrival_time=self.now, dc_name=None)
        # route
        dc_name, Lnet, path, score = select_dc(ing, jtype, self.dcs, self.coeffs_map,
                                               self.graph, self.carbon, self.router_policy)
        # simple transfer time model
        data_gb = 0.05 if jtype == 'inference' else 5.0
        # bottleneck not returned here to keep simple; you can extend Graph to return it.
        transfer_s = Lnet  # + data_gb / max(1e-6, bottleneck)  # bật khi có capacity
        # schedule transfer completion
        self._schedule(self.now + transfer_s, 'xfer_done',
                       {'ing': ing_name, 'dc': dc_name, 'jid': jid, 'job': job, 'net_lat_s': Lnet})

        # schedule next arrival at this ingress
        ia = (self.arr_inf if jtype=='inference' else self.arr_trn).next_interarrival(self.now)
        self._schedule(self.now + ia,
                       'arrival_inf' if jtype=='inference' else 'arrival_trn',
                       {'ing': ing_name})

    def _sample_job_size(self, jtype: str) -> float:
        import math, random
        if jtype == 'inference':
            xm, alpha = 0.02, 2.5
            u = max(1e-9, 1 - random.random())
            return xm / (u ** (1/alpha))
        mu, sigma = math.log(3.0), 0.6
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

        with open(self.job_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                job.jid, job.dc_name, job.jtype, f"{job.size:.4f}", dc.name,
                f"{f_used:.3f}", g, f"{getattr(job,'net_latency_s',0.0):.4f}",
                f"{job.start_time:.6f}", f"{job.finish_time:.6f}",
                f"{(job.finish_time - job.start_time):.6f}",
                f"{T_pred:.6f}", f"{P_pred:.2f}", f"{E_pred:.2f}"
            ])

        # Bandit: cập nhật reward = - energy_per_unit (hoặc đổi sang carbon/cost nếu muốn)
        if getattr(self, "bandit", None) is not None:
            self.bandit.update(dc.name, job.jtype, f_used, E_pred)

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
                run_inf = sum(1 for j,_ in dc.running_jobs.values() if j.jtype == 'inference')
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

    def best_nf_grid(n_max: int, freq_levels,
                     p_coeffs: TrainPowerCoeffs, t_coeffs: TrainLatencyCoeffs,
                     objective: str = "energy",  # "energy" | "carbon"
                     carbon_intensity: float = 0.0,  # gCO2/kWh (tương đối cũng được)
                     deadline_s: float | None = None):
        """
        Trả về (n*, f*, T*, P*, E*). f_levels là list float (đã chuẩn hóa như đang dùng).
        """
        best = None
        for n in range(1, max(1, int(n_max)) + 1):
            for f in freq_levels:
                T = step_time_s(n, f, t_coeffs)  # s per unit
                P = n * gpu_power_w(f, p_coeffs)  # W
                E = P * T  # J per unit
                if deadline_s is not None and T > deadline_s:
                    continue
                score = E if objective == "energy" else (E * carbon_intensity)
                cand = (score, n, f, T, P, E)
                if (best is None) or (cand[0] < best[0]):
                    best = cand
        if best is None:
            # fallback: dùng n=1, f=max
            fmax = max(freq_levels)
            T = step_time_s(1, fmax, t_coeffs);
            P = gpu_power_w(fmax, p_coeffs);
            E = P * T
            return (1, fmax, T, P, E)
        _, n, f, T, P, E = best
        return (n, f, T, P, E)

    def _start_job_with_nf(self, dc, job, n, f):
        # cấp phát
        n = max(1, min(n, dc.free_gpus))
        if n <= 0:
            (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job);
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
