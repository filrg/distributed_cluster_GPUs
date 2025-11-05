# Configuration

## f

```python
@dataclass
class GPUType:
    name: str        # Ex: "A100-SXM4", "H100-PCIe"
    p_idle: float    # W — power of one GPU when idle (P0/P8 depending)
    p_peak: float    # W — “peak” power used by baseline model for dynamic component
    p_sleep: float   # W — power when GPU power-gated/sleep.
    alpha: float     # exponent factor for DVFS in baseline model (P ~ f^alpha)
```

* **`name`**
  Specify the **hardware variants** (SXM/PCIe, HBM capacity) for better power consumption accuracy. Examples: `"A100-SXM4"`, `"H100-PCIe"`.

* **`p_idle` (W)**
  Power of **one GPU** turning on but **not running** (no kernels). In baseline, when the GPU is “busy” we **add** the dynamic component, so `p_idle` always exists.
  Note:

  * Depend on driver/firmware, ECC, MIG, and “persistence mode”.
  * No standard numbers published by the manufacturer.

* **`p_peak` (W)**
  In the **baseline model** (file `simcore/energy.py`), the power when the GPU is active at the nominal frequency:

  ```
  P_active ≈ p_idle + p_peak * (f ** alpha)
  ```

  Here `p_peak` is the **dynamic component**, does not necessarily match TDP/TBP on datasheet.

* **`p_sleep` (W)**
  Power when **power-gating** is active (DC has `power_gating=True`), meaning part of the hardware is "off" or in deep sleep mode (P8). Used to simulate power-saving when offloading and turning off GPUs.

* **`alpha`**
  Exponent factor for DVFS in baseline. Values between 2–3.5 is reasonable; default 3.0.

**Example configuration:**

```python
A100_SXM  = GPUType("A100-SXM4",  p_idle=50.0, p_peak=(400.0), p_sleep=30.0, alpha=3.0)
H100_PCIe = GPUType("H100-PCIe",  p_idle=45.0, p_peak=(350.0), p_sleep=28.0, alpha=3.0)
L4        = GPUType("L4",         p_idle=15.0, p_peak=(72.0),  p_sleep=8.0,  alpha=3.0)
```

## Power consumption and Time profile

### `TrainPowerCoeffs(α_p, β_p, γ_p)` — Power model based on frequency

For each **GPU**:

$$
P_{\text{gpu}}(f) = \alpha_p f^3 + \beta_p f + \gamma_p \quad\text{(W/GPU)}
$$

* `f` is **normalized frequency** (ví dụ các mức `freq_levels = [0.5, 0.8, 1.0]`). If using **GHz scale**, the coefficients must be refitted accordingly.
* `α_p` (W) — cubic **dynamic** component on f (for DVFS).
* `β_p` (W) — linear dynamic component on f (leak + linear part).
* `γ_p` (W) — **static offset** close to “idle offset” when job is running (different from `p_idle` of GPUType, as this is offset **within** the job model).

Power consumption for **one job** using `n` GPU at frequency `f`:

$$
P_{\text{job}}(n,f)=n \cdot P_{\text{gpu}}(f)
$$

**Fitting suggestion:**
- Measure power consumption at different frequency levels when the job is running (stabilized for several seconds) → polynomial regression $(f^3, f, 1)$.
- Constraints: $\alpha_p\ge0, \beta_p\ge0, \gamma_p\ge 0$.

### `TrainLatencyCoeffs(α_t, β_t, γ_t)` — Time model for each "job unit"

For $n=1$:
$$
T(n,f) = \alpha_t + \frac{\beta_t}{f} \quad \text{(s / unit)}
$$
For $n>1$:
$$
T(n,f) = (\alpha_t + \frac{\beta_t}{f} + \gamma_t n) / n \quad \text{(s / unit)}
$$

* `unit` is the **job unit** defined in the simulator (e.g., “one step training”, “one micro-batch”, or “one batch inference of size b”).
* `α_t` (s/unit) — fix overhead (IO, setup).
* `β_t` (s·(unit)·f) — part that **inversely proportional** to f (increasing f makes it faster).
* `γ_t` (s/(unit·GPU)) — **scaling penalty** by number of GPUs (synchronization, all-reduce…). Should be small, else will negate scale-out benefit.


### How simulator uses coefficients

* **Service time** of one job:
  `service_time = job.size * step_time_s` (in code: `job.size` is the number of unit).
* **Instantaneous power** when job is running:
  `P_job = n * P_gpu(f)`.
* **Predicted energy** (log/comparision):
  `E_pred = P_job * T(n,f)`.

Coefficients **do not depend** on `GPUType.p_idle/p_sleep` — these only used for **idle GPU** (idle/sleep) in DC level. When a job is running, its power comes from `TrainPowerCoeffs`.


### Units & Recommended Value Ranges

* `f`
  * **Recommend to normalized values**: $f\in(0,1]$, with 1.0 = “nominal”.
* `P` (W), `T` (s/unit).
* Constraints:

  * $\alpha_p,\beta_p,\gamma_p \ge 0$.
  * $\alpha_t,\beta_t \ge 0$; $\gamma_t \ge 0$ but **small**.
  * Ensure $T(n,f)>0$ at all usage levels.


## Request/Arrival

Code:
```python
build_arrivals(
  inf_mode=args.inf_mode,   # "poisson" | "sinusoid" | "off"
  inf_rate=args.inf_rate,   # float (requests/second)
  inf_amp=args.inf_amp,     # float, amplitude (for sinusoid)
  inf_period=args.inf_period,# float, period (seconds, for sinusoid)
  trn_mode=args.trn_mode,   # "poisson" | "sinusoid"
  trn_rate=args.trn_rate    # float, requests/second
)
```

Declare arrival process:

* `inference` (short, SLA-sensitive)
* `training` (long, less SLA-sensitive, preemptible)

The simulator will generate **arrival events** based on this and push them to the router/DC.

**Arguments**

* `*_mode`: type of arrival
	* `"off"`: turn off train/inference.
    * `"poisson"`: **constant** arrival rate $\lambda$. Inter-arrival time is i.i.d. **exponential** ⇒ CV=1.
	* `"sinusoid"`: arrival rate **varies by time**:

$$
\lambda(t)=\max\big(0,\ \text{rate}\cdot[1 + \text{amp}\cdot \sin(2\pi\, t/\text{period})]\big)
$$

  Using **thinning** to generate arrival; amplitude is controlled by `amp`.

> If the workload is steady, use Poisson; if there's a day/night cycle, peak hours, other events ..., use Sinusoid.

* `*_rate` (unit: requests/s)
	* **Average speed** of the process.
	* With sinusoid, **expected value** is still `rate` (average of sine is 0). Don't confuse `amp` as changing the total load—it only **redistributes over time**.

* `inf_amp` (0…∞, only sinusoid): **Relative amplitude**.

  * `0` ⇒ no oscillation (constant).
  * `0.6` ⇒ $\lambda(t)$ oscillates ±60% around `rate`.
  * > 1 is allowed, but the negative part will be **clipped to 0**. Avoid >1 else “cut” low phases.

* `inf_period` (seconds, only sinusoid). Oscillation period. Examples:

  * `3600` ⇒ **hourly**,
  * `86400` ⇒ **day/night**.

* `trn_rate`, `trn_mode`. Similar usage to inference but for **training**. Usually Poisson with low arrival rate (`0.01–0.05 req/s`) to reflect heavy jobs that appear infrequently.

## Generating arrivals

* **Poisson**: $\Delta \sim \text{Exp}(\lambda)$. If `rate<=0` ⇒ **no arrival** (return `inf`).
* **Sinusoid**: use **acceptance-rejection (thinning)** với $\lambda_{\max}=\text{rate}(1+|\text{amp}|)$. Generate inter-arrival time ~ Exp($\lambda_{\max}$), then **accept** with probability $\lambda(t)/\lambda_{\max}$.

* **Unit**: **requests/s**, **period in seconds**.
* **Load balancing**: total $\mathbb{E}[N]$ in time $T$ is `rate*T` for sinusoid. If the target arrival isn’t met in the log, the error lies in the service time ($T(n,f)$) or router, not in `amp`.
* **Burst**: Poisson **does not** create long “wave”; if prefer high throughput instead of random bursts, use sinusoid or load **real trace**.


### Estimating parameters

If you want inference **no overloading**:

* Each GPU at `f=1` can handle about $\mu$ **unit/s** (from `TrainLatencyCoeffs`), an inference job consumes $u$ unit on average.
* Each DC has $G$ GPU, allocates $g$ GPU for each request inference on average.
  ⇒ Capacity $\text{cap} \approx \dfrac{G}{g}\cdot\dfrac{\mu}{u}$ **req/s**.
* Set `inf_rate` < **total** `cap` of the DCs *that router will choose*; if using sinusoid, ensure **peak** $\text{rate}(1+\text{amp})$ ≤ **total cap** or accept queue.

### Arrival config

1) Inference day/night cycle, training steady

```bash
python run_sim_paper.py \
  --inf-mode sinusoid --inf-rate 4.0 --inf-amp 0.7 --inf-period 86400 \
  --trn-mode poisson  --trn-rate 0.02 \
  --duration 72000
```

2) High workload during peak hours every 5 minutes, training steady

```bash
python run_sim_paper.py \
  --inf-mode sinusoid --inf-rate 6.0 --inf-amp 0.6 --inf-period 300 \
  --trn-mode poisson  --trn-rate 0.03 \
  --duration 1800
```

3) Poisson to benchmark policy

```bash
python run_sim_paper.py \
  --inf-mode poisson --inf-rate 5.0 \
  --trn-mode poisson --trn-rate 0.02 \
  --duration 1200
```

# Algorithms
List of algorithms implemented as baseline methods:
- `joint_nf`: SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving.
- `bandit`: https://tor-lattimore.com/downloads/book/book.pdf

## `Default Policy` (DP)

No power control; only default policy allocates GPU and sets one fixed $f$ upon DC.

```bash
python run_sim_paper.py --algo default_policy \
  --duration 1200 --log-interval 5
```

Expected: `cluster_log.csv` shows that `freq` stays the same; power only changes by number of jobs.

Required turning on **per-job DVFS + reschedule**.

## `joint_nf` (JOINT-NF)

Upon arrival, conduct a grid search $n \in [1 .. N_{\max}], f \in \text{freq\_levels}$; choose **min energy** configuration (or carbon/cost) which **satisfies SLA**.

```bash
python run_sim_paper.py --algo joint_nf \
  --duration 1200 --log-interval 5
```

Suggest:

* Assign `job.deadline` for inference if SLO constraints.
* Coefficients $T(n,f)$ should be acceptable.
  As DVFS-aware, SLO-aware scaling on LLM shows a "sweet spot" (e.g., \~1050 MHz on A100) brings +\~37% energy efficiency with negligible impact on performance.

## `bandit` (UCB1)

Each DC/job_type treats each $f$ as an arm. Upon arrival: **select** $f$ using UCB1; when the job finishes: **update** reward = −(energy per unit). No need for an exact model, it can adapt to the workload.

```bash
python run_sim_paper.py --algo bandit \
  --duration 1200 --log-interval 5
```

Suggest:

* Set `init_explore=1` (default in learner) to explore all $f$.
* Reward focus can be **carbon** (−E·CI) or **cost** (−E·price).

## Our algorithm `CHSAC-AF`

```bash
python3 run_sim_paper.py --algo chsac_af --upgr-buffer 200000 --upgr-batch 256 --upgr-warmup 1000 --upgr-device cuda --sla_p99_ms 500.0 --inf-mode off --trn-rate 0.02 --duration 604800 --log-interval 20 --log-path test_run_0.02/CHSAC-AF
```

## Plot

Each run (by config/algorithm) is placed in a folder that consists of:

* `cluster_log.csv`
* `job_log.csv`

Comparing algorithms
```bash
python plot_sim_result.py --run baseline=./runs/baseline --run cap_greedy=./runs/cap_greedy --run carbon=./runs/carbon_cost --outdir ./figs --bin 5
```
Single algorithm 
```bash
python plot_single_algo.py --run baseline=./runs/baseline --run cap_greedy=./runs/cap_greedy --run carbon=./runs/carbon_cost --outdir ./debug_figs --bin 5
```

* `NAME=DIR`: The name to display on the legend and the folder containing the CSV. For example: baseline=./runs/baseline
* `--outdir`: Directory to save the plots.
* `--bin`: The bin size (in seconds) for the throughput chart.
* `--scaledown`: The step size when reading rows from the log.
* `--pdf`: Save the plot as PDF.