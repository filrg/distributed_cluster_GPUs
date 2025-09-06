import csv, heapq, itertools, random
from typing import Dict, List, Optional, Tuple
from .models import Job, DataCenter
from .arrivals import ArrivalConfig
from .policy import PolicyConfig, select_gpus_and_set_freq

Event = Tuple[float, int, str, dict]

class Simulator:
    def __init__(self, dcs: Dict[str, DataCenter],
                 arrival_inf: ArrivalConfig,
                 arrival_train: ArrivalConfig,
                 policy: PolicyConfig,
                 sim_duration: float = 3600.0,
                 log_interval: float = 10.0,
                 rng_seed: int = 42):
        self.now = 0.0
        self.end_time = sim_duration
        self.event_q: List[Event] = []
        self.seq = itertools.count()
        self.policy = policy
        self.dcs = dcs
        self.arr_inf = arrival_inf
        self.arr_trn = arrival_train
        random.seed(rng_seed)
        self.jid_counter = itertools.count(1)
        self.jobs: Dict[int, Job] = {}

        self.cluster_log_path = "cluster_log.csv"
        self.job_log_path = "job_log.csv"

        self._schedule(self.now + self.arr_inf.next_interarrival(self.now), 'arrival_inf', {})
        self._schedule(self.now + self.arr_trn.next_interarrival(self.now), 'arrival_trn', {})
        self._schedule(self.now + log_interval, 'log', {'interval': log_interval})

    def _schedule(self, t: float, etype: str, payload: dict):
        if t == float('inf') or t > self.end_time + 1e-9:
            return
        heapq.heappush(self.event_q, (t, next(self.seq), etype, payload))

    def _pop(self):
        return heapq.heappop(self.event_q) if self.event_q else None

    def run(self):
        with open(self.cluster_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["time_s","dc","freq","busy","free","q_inf","q_train","power_W","energy_kJ"])
        with open(self.job_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["jid","type","size","arrival_s","dc","gpus","start_s","finish_s","latency_s"])

        while self.event_q:
            ev = self._pop()
            if ev is None: break
            t,_,etype,payload = ev
            if t > self.end_time: break

            for dc in self.dcs.values():
                dc.accrue_energy(t)

            self.now = t
            if etype == 'arrival_inf':
                self.handle_arrival('inference')
            elif etype == 'arrival_trn':
                self.handle_arrival('training')
            elif etype == 'job_finish':
                self.handle_job_finish(payload['dc'], payload['jid'])
            elif etype == 'log':
                self.handle_log(payload['interval'])
            else:
                raise RuntimeError(f"Unknown event {etype}")

        for dc in self.dcs.values():
            dc.accrue_energy(self.end_time)

    def handle_arrival(self, jtype: str):
        jid = next(self.jid_counter)
        size = self.sample_job_size(jtype)
        deadline = self.now + 0.5 if jtype == 'inference' else None
        job = Job(jid=jid, jtype=jtype, size=size, arrival_time=self.now, deadline=deadline)
        self.jobs[jid] = job
        dc_name = self.choose_dc_for_job(job)
        job.dc_name = dc_name
        dc = self.dcs[dc_name]
        self.enqueue_or_start(dc, job)
        ia = (self.arr_inf if jtype=='inference' else self.arr_trn).next_interarrival(self.now)
        self._schedule(self.now + ia, 'arrival_inf' if jtype=='inference' else 'arrival_trn', {})

    def sample_job_size(self, jtype: str) -> float:
        import math, random
        if jtype == 'inference':
            xm, alpha = 0.02, 2.5
            u = max(1e-9, 1 - random.random())
            return xm / (u ** (1/alpha))
        else:
            mu, sigma = math.log(3.0), 0.6
            return max(0.1, random.lognormvariate(mu, sigma))

    def choose_dc_for_job(self, job: Job) -> str:
        best_dc, best_metric = None, float('inf')
        for name, dc in self.dcs.items():
            f = dc.current_freq
            gt = dc.gpu_type
            p_active = gt.p_idle + gt.p_peak * (f ** gt.alpha)
            metric = p_active / max(1e-6, f)
            metric *= (1.0 + 0.1 * (len(dc.q_inf) + len(dc.q_train)))
            if metric < best_metric:
                best_metric, best_dc = metric, name
        return best_dc

    def enqueue_or_start(self, dc: DataCenter, job: Job):
        if dc.free_gpus > 0:
            gpus = select_gpus_and_set_freq(dc, job, self.policy)
            if gpus > 0:
                return self.start_job(dc, job, gpus)
        (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job)

    def start_job(self, dc: DataCenter, job: Job, gpus: int):
        if gpus <= 0:
            (dc.q_inf if job.jtype == 'inference' else dc.q_train).append(job)
            return
        dc.busy_gpus += gpus
        dc.running_jobs[job.jid] = (job, gpus)
        job.gpus_assigned = gpus
        job.start_time = self.now
        f = max(1e-6, dc.current_freq)
        service_time = job.size / (gpus * f)
        self._schedule(self.now + service_time, 'job_finish', {'dc': dc.name, 'jid': job.jid})

    def handle_job_finish(self, dc_name: str, jid: int):
        dc = self.dcs[dc_name]
        tup = dc.running_jobs.pop(jid, None)
        if not tup: return
        job, g = tup
        dc.busy_gpus = max(0, dc.busy_gpus - g)
        job.finish_time = self.now
        with open(self.job_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([job.jid, job.jtype, f"{job.size:.4f}", f"{job.arrival_time:.6f}",
                                    job.dc_name, g, f"{job.start_time:.6f}", f"{job.finish_time:.6f}",
                                    f"{(job.finish_time - job.arrival_time):.6f}"])
        while dc.free_gpus > 0:
            nxt = None
            if self.policy.inf_priority and dc.q_inf:
                nxt = dc.q_inf.pop(0)
            elif dc.q_train:
                nxt = dc.q_train.pop(0)
            if nxt is None: break
            gpus = select_gpus_and_set_freq(dc, nxt, self.policy)
            if gpus <= 0:
                (dc.q_inf if nxt.jtype == 'inference' else dc.q_train).insert(0, nxt)
                break
            self.start_job(dc, nxt, gpus)

    def handle_log(self, interval: float):
        with open(self.cluster_log_path, 'a', newline='') as f:
            w = csv.writer(f)
            for name, dc in self.dcs.items():
                w.writerow([f"{self.now:.3f}", name, f"{dc.current_freq:.2f}",
                            dc.busy_gpus, dc.free_gpus, len(dc.q_inf), len(dc.q_train),
                            f"{dc.instantaneous_power_w():.2f}", f"{dc.energy_joules/1000.0:.4f}"])
        self._schedule(self.now + interval, 'log', {'interval': interval})
