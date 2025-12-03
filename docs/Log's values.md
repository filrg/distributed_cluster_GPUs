# Values in the log files

## `cluster_log.csv`

* **time\_s** — simulation time (seconds) at the logging instant.
* **dc** — data center name.
* **freq** — frequency level *set at the DC level* (`dc.current_freq`, normalized to 0–1).  
  If **per-job DVFS** is enabled, each job can run at `job.f_used` ≠ `freq`. See `f_used` in `job_log.csv` for details.
* **busy** — number of busy GPU (running jobs) in DC.
* **free** — number ò idle GPU = `total_gpus − busy`.
* **run\_total** — number of job running in DC.
* **run\_inf** — number of running inference job.
* **run\_train** — number of running training job.
* **q\_inf** — queue length of inference jobs at DC.
* **q\_train** — queue length of training jobs at DC.
* **util\_inst** — instantaneous GPU occupancy = `busy / total_gpus` (0..1).
* **util\_avg** — average GPU occupancy from simulation's start until `time_s`:

$$
\textstyle util\_avg = \frac{\sum \text{busy}(t)\cdot \Delta t}{\text{total\_gpus}\cdot (time\_s - t_{start})}
$$
* **acc_job_unit** — **cumulative job_size unit** of DC, overtime at `log` events (by `dc.running_jobs`) and `job_finish` events.
* **power\_W** — **Instantaneous power** of DC (W), using paper's model:

$$
\textstyle P_{\text{DC}}(t) = \sum_{\text{job }j} n_j \cdot P_{\text{gpu}}(f_j)\;+\;(\text{free})\cdot
\begin{cases}
p_{\text{sleep}}, & \text{if power_gating}\\
p_{\text{idle}}, & \text{otherwise}
\end{cases}
$$

with $f_j = job.f\_used$ (if defined) or DC's`freq`.
* **energy\_kJ** — **cumulative energy** of DC from simulation's start until `time_s` (kJ):

$$
\textstyle E(t) = \int_0^{t} P_{\text{DC}}(\tau)\,d\tau \quad (\text{J→kJ})
$$

## `job_log.csv`

* **jid** — job id.
* **ingress** — ingress node destination.
* **type** — job type: `inference` | `training`.
* **size** — number of “work units” of job. Service time scale by `size`.
* **dc** — data center that execute the job.
* **f\_used** — frequency used for the job (0–1). If per-job DVFS, this is `job.f_used`; else `dc.current_freq` from the start.
* **n\_gpus** — number of GPUs allocated for the job (running parallelly).
* **net\_lat\_s** (s) — network (WAN) latency before it reaches DC (transfer).
* **start\_s**, **finish\_s** (s) — simulation timestamps when the job starts/finishes compute.
* **latency\_s** (s) — actual latency compute = `finish_s − start_s` (without network, queue latency).
* **preempt_count** (times): number of times the (training) job is preempted.
* **T\_pred** (s/unit) — predicted compute time **by unit**: $T(n,f)$ from `TrainLatencyCoeffs`.
* **P\_pred** (W) — predicted **job's power** while running: $P(n,f)=n\cdot P_{\text{gpu}}(f)$ from `TrainPowerCoeffs`.
* **E\_pred** (J/unit) — predicted energy **by unit** of job: $E_{\text{unit}}=P(n,f)\cdot T(n,f)$.
