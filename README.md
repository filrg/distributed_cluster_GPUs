# Cấu hình

## Các loại GPU

```python
@dataclass
class GPUType:
    name: str        # ví dụ: "A100-SXM4", "H100-PCIe"
    p_idle: float    # W — công suất 1 GPU khi ở trạng thái rỗi (P0/P8 tuỳ cấu hình)
    p_peak: float    # W — “đỉnh” mà mô hình baseline dùng cho phần động
    p_sleep: float   # W — công suất khi GPU bị gate/ngủ (power-gating), không nhận job
    alpha: float     # hệ số mũ cho DVFS trong mô hình baseline (P ~ f^alpha)
```

* **`name`**
  Ghi rõ **biến thể phần cứng** (SXM/PCIe, dung lượng HBM), vì công suất khác nhau. Ví dụ: `"A100-SXM4"`, `"H100-PCIe"`. Đừng ghi chung chung rồi kỳ vọng độ chính xác.

* **`p_idle` (W)**
  Công suất **một GPU** đang bật nhưng **không chạy** (no kernels). Trong baseline, khi GPU “bận” ta **cộng thêm** thành phần động, nên `p_idle` là nền tảng luôn tồn tại.
  Note:

  * Phụ thuộc driver/firmware, ECC, MIG, và “persistence mode”.
  * Không có con số hãng công bố chuẩn.

* **`p_peak` (W)**
  Trong **mô hình baseline** (file `simcore/energy.py`), công suất khi GPU hoạt động ở tần số chuẩn được tính:

  ```
  P_active ≈ p_idle + p_peak * (f ** alpha)
  ```

  Ở đây `p_peak` là **biên độ phần động**, không nhất thiết đúng bằng TDP/TBP trên datasheet.

* **`p_sleep` (W)**
  Công suất khi **power-gating** (DC có `power_gating=True`), tức “tắt” một phần phần cứng/đưa về P8 sâu. Dùng để mô phỏng tiết kiệm khi dồn tải và tắt bớt GPU.

* **`alpha`**
  Hệ số mũ cho DVFS trong baseline. Giá trị 2–3.5 thường hợp lý; 3.0 là mặc định.

**Ví dụ cấu hình**

```python
A100_SXM  = GPUType("A100-SXM4",  p_idle=50.0, p_peak=(400.0), p_sleep=30.0, alpha=3.0)
H100_PCIe = GPUType("H100-PCIe",  p_idle=45.0, p_peak=(350.0), p_sleep=28.0, alpha=3.0)
L4        = GPUType("L4",         p_idle=15.0, p_peak=(72.0),  p_sleep=8.0,  alpha=3.0)
```

## Profile tiêu thụ công suất và thời gian

### `TrainPowerCoeffs(α_p, β_p, γ_p)` — mô hình công suất theo tần số

Công thức (mỗi **GPU**):

$$
P_{\text{gpu}}(f) = \alpha_p f^3 + \beta_p f + \gamma_p \quad\text{(W/GPU)}
$$

* `f` là **tần số chuẩn hoá** (ví dụ các mức `freq_levels = [0.5, 0.8, 1.0]`). Nếu dùng **GHz thực**, phải fit lại hệ số cho đúng đơn vị.
* `α_p` (W) — thành phần **động** bậc ba theo f (phù hợp DVFS).
* `β_p` (W) — thành phần động bậc một theo f (rò rỉ + phần tuyến tính).
* `γ_p` (W) — **bù tĩnh** gần tương đương “idle offset” khi job đang chạy (khác `p_idle` của GPUType, vì đây là offset **trong** mô hình job).

Điện năng cho **một job** dùng `n` GPU tại tần số `f`:

$$
P_{\text{job}}(n,f)=n \cdot P_{\text{gpu}}(f)
$$

**Gợi ý fit:**

* Đo công suất theo nhiều mức f khi job chạy (ổn định vài giây) → hồi quy đa thức $(f^3, f, 1)$.
* Ràng buộc hợp lý: $\alpha_p\ge0, \beta_p\ge0, \gamma_p\ge 0$.

### `TrainLatencyCoeffs(α_t, β_t, γ_t)` — mô hình thời gian trên mỗi “đơn vị công việc”

Công thức:

Với $n=1$:
$$
T(n,f) = \alpha_t + \frac{\beta_t}{f} \quad \text{(s / unit)}
$$
Với $n>1$:
$$
T(n,f) = (\alpha_t + \frac{\beta_t}{f} + \gamma_t n) / n \quad \text{(s / unit)}
$$

* `unit` là **đơn vị công việc** người dùng định nghĩa trong simulator (ví dụ “một step training”, “một micro-batch”, hay “một batch inference cỡ b”).
* `α_t` (s/unit) — overhead cố định (IO, setup).
* `β_t` (s·(unit)·f) — phần tính toán **ngược tỉ lệ** với f (tăng f thì nhanh hơn).
* `γ_t` (s/(unit·GPU)) — **phạt mở rộng** theo số GPU (đồng bộ, all-reduce…). Giá trị này **nhỏ**; nếu đặt lớn sẽ “giết” mọi lợi ích scale-out.

### Job size
* Inference: Phân phối Pareto với `xm` ~ scale tối thiểu, `alpha` ~ shape parameter (quyết định độ heavy-tailed).
  * Theo lý thuyết, $\alpha \le 2$ thì Variance sẽ tiến đến vô cực.

$$
E[X] = \frac{\alpha x_m}{\alpha - 1}, \quad \alpha > 1
$$

$$
Var[X] = \frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)}, \quad \alpha > 2
$$
* Training: $X \sim \text{LogNormal}(\mu, \sigma^2)$

$$
E[X] = e^{\mu + \sigma^2 / 2}
$$

$$
Var[X] = (e^{\sigma^2} - 1) \, e^{2\mu + \sigma^2}
$$
Tùy chỉnh các tham số để kiểm soát `service_time` hợp lý.

### Cách simulator dùng các hệ số

* **Service time** của một job:
  `service_time = job.size * T(n,f)` (trong code: `job.size` là số unit).
* **Công suất tức thời** khi job chạy:
  `P_job = n * P_gpu(f)`.
* **Năng lượng dự đoán** (tiện log/so sánh):
  `E_pred = P_job * T(n,f)`.

Các hệ số **không phụ thuộc** `GPUType.p_idle/p_sleep` — các số này chỉ dùng cho **GPU rảnh** (idle/sleep) ở mô hình DC. Khi job chạy, điện năng job đến từ `TrainPowerCoeffs`.

### Đơn vị & miền giá trị khuyến nghị

* `f`

  * **Khuyên dùng với giá trị chuẩn hoá**: $f\in(0,1]$, với 1.0 = mức “nominal”.
* `P` (W), `T` (s/unit).
* Ràng buộc:

  * $\alpha_p,\beta_p,\gamma_p \ge 0$.
  * $\alpha_t,\beta_t \ge 0$; $\gamma_t \ge 0$ nhưng **nhỏ**.
  * Đảm bảo $T(n,f)>0$ ở mọi mức dùng.

## Request đến

Cấu trúc code:
```python
build_arrivals(
  inf_mode=args.inf_mode,   # "poisson" | "sinusoid" | "off"
  inf_rate=args.inf_rate,   # float, đơn vị: yêu cầu/giây
  inf_amp=args.inf_amp,     # float, biên độ dao động (chỉ dùng khi sinusoid)
  inf_period=args.inf_period,# float, chu kỳ (giây, chỉ dùng khi sinusoid)
  trn_mode=args.trn_mode,   # "poisson" | "sinusoid"
  trn_rate=args.trn_rate    # float, yêu cầu/giây
)
```

Khai báo **quy luật đến** (arrival process) cho hai luồng:

* `inference` (ngắn, nhạy SLA)
* `training` (dài, ít nhạy SLA, có thể bị preempt)

Simulator sẽ sinh **sự kiện đến** theo cấu hình này và đẩy vào router/DC.

**Các tham số**

* `*_mode`: kiểu tiến trình đến
	* `"off"`: dùng để tắt train/inference nếu cần.
    * `"poisson"`: tốc độ đến **không đổi** $\lambda$. Khoảng cách giữa hai arrival i.i.d. **mũ** ⇒ CV=1.
	* `"sinusoid"`: tốc độ đến **thay đổi theo thời gian**:

$$
\lambda(t)=\max\big(0,\ \text{rate}\cdot[1 + \text{amp}\cdot \sin(2\pi\, t/\text{period})]\big)
$$

  Dùng **thinning** để sinh arrival; biên dao động được kiểm soát bởi `amp`.

> Nếu workload phẳng lì thì cứ Poisson; còn đã có nhịp ngày/đêm, giờ cao điểm, sự kiện… thì Sinusoid.

* `*_rate` (đơn vị: yêu cầu/giây)
	* **Tốc độ trung bình** của tiến trình.
	* Với sinusoid, **kỳ vọng** vẫn bằng `rate` (vì trung bình của sin bằng 0). Đừng ngộ nhận `amp` làm đổi tổng tải trung bình—nó chỉ **phân phối lại theo thời gian**.

* `inf_amp` (0…∞, chỉ sinusoid): **Biên độ tương đối**.

  * `0` ⇒ không dao động (quay về hằng).
  * `0.6` ⇒ $\lambda(t)$ dao động ±60% quanh `rate`.
  * > 1 vẫn được, phần âm sẽ bị **cắt về 0**. Nhưng hạn chế >1 nếu không muốn “cụt” pha thấp.

* `inf_period` (giây, chỉ sinusoid). Chu kỳ của dao động. Ví dụ:

  * `3600` ⇒ nhịp theo **giờ**,
  * `86400` ⇒ **ngày/đêm**.

* `trn_rate`, `trn_mode`. Tương tự phía inference nhưng cho **luồng training**. Thường để Poisson với tốc độ thấp (ví dụ `0.1–0.5 req/s`) để phản ánh job dài ít xuất hiện.

## Cách simulator sinh arrival

* **Poisson**: $\Delta \sim \text{Exp}(\lambda)$. Nếu `rate<=0` ⇒ **không có arrival** (trả `inf`).
* **Sinusoid**: dùng **acceptance-rejection (thinning)** với $\lambda_{\max}=\text{rate}(1+|\text{amp}|)$. Sinh thời gian chờ từ Exp($\lambda_{\max}$), rồi **chấp nhận** với xác suất $\lambda(t)/\lambda_{\max}$.

* **Đơn vị**: mọi thứ đang là **yêu cầu/giây**, **chu kỳ theo giây**.
* **Cân bằng tải**: tổng $\mathbb{E}[N]$ trong thời gian $T$ là `rate*T` cho sinusoid. Nếu không đạt target arrival trong log ⇒ lỗi nằm ở **service time** (T(n,f)) hoặc **router**, **không** phải ở `amp`.
* **Burst**: Poisson **không** tạo “wave” dài; nếu cần lưu lượng lớn thay vì nhiễu lẻ tẻ, dùng sinusoid hoặc nạp **trace thực**.


### Công thức “ước lượng” để set tham số hợp lý

Giả sử muốn inference **không quá tải**:

* Mỗi GPU ở `f=1` xử lý được khoảng $\mu$ **unit/s** (từ `TrainLatencyCoeffs`), một job inference tiêu tốn $u$ unit trung bình.
* Một DC có $G$ GPU, phân bổ trung bình $g$ GPU cho mỗi request inference.
  ⇒ Năng lực gần bằng $\text{cap} \approx \dfrac{G}{g}\cdot\dfrac{\mu}{u}$ **req/s**.
* Chọn `inf_rate` < **tổng** `cap` của các DC *mà router sẽ chọn tới*; nếu dùng sinusoid, đảm bảo **đỉnh** $\text{rate}(1+\text{amp})$ vẫn ≤ **tổng cap** hoặc chấp nhận hàng đợi.

### Mẫu cấu hình

1) Inference nhịp ngày/đêm, training lác đác

```bash
python run_sim_paper.py \
  --inf-mode sinusoid --inf-rate 4.0 --inf-amp 0.7 --inf-period 86400 \
  --trn-mode poisson  --trn-rate 0.2 \
  --duration 72000
```

2) Căng tải giờ cao điểm 5 phút/lần, training êm

```bash
python run_sim_paper.py \
  --inf-mode sinusoid --inf-rate 6.0 --inf-amp 0.6 --inf-period 300 \
  --trn-mode poisson  --trn-rate 0.3 \
  --duration 1800
```

3) Poisson phẳng để benchmark policy

```bash
python run_sim_paper.py \
  --inf-mode poisson --inf-rate 5.0 \
  --trn-mode poisson --trn-rate 0.2 \
  --duration 1200
```

# Cách chạy với các thuật toán khác nhau

Danh sách các mode thuật toán được dựng từ các nghiên cứu:
- `cap_uniform`, `cap_greedy`: Providing Load Flexiblity by Reshaping Power Profiles of Large Language Models.
- `joint_nf`: SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving.
- `bandit`: https://tor-lattimore.com/downloads/book/book.pdf
- `carbon_cost`: Carbon-Aware Computing for Datacenters.

## `baseline` (không hãm công suất)

Không điều khiển nguồn; chỉ policy mặc định cấp phát GPU và ấn định một $f$ chung cho DC.

```bash
python run_sim_paper.py --algo baseline \
  --duration 1200 --log-interval 5
```

Kỳ vọng: `cluster_log.csv` cho thấy `freq` giữ nguyên; công suất chỉ thay đổi theo số job.

## `cap_uniform` (power-capping đồng loạt theo DC)

Giảm $f$ từng nấc; mỗi vòng chọn **DC** có $\Delta P$ lớn nhất khi hạ một nấc.

```bash
python run_sim_paper.py --algo cap_uniform \
  --duration 1200 --log-interval 5 --power-cap 8000 --control-interval 2 
```

Kỳ vọng: tổng $P$ tiệm cận `--power-cap`. Nếu vượt cap không giảm tiếp, kiểm tra `freq_levels` và còn “nấc” để hạ không. (Cơ chế chọn theo bước DVFS dựa trên ý tưởng điểm biên.

## `cap_greedy` (aggregate-based, cấp **job**)

Xây atoms per-job (bước $f_i\to f_{i-1}$ với $\Delta P,\Delta V$), sắp theo $\Delta P/\Delta V$, rồi **đánh vào job rẻ nhất** cho tới khi bù xong thâm hụt so với cap.

```bash
python run_sim_paper.py --algo cap_greedy \
  --duration 1200 --log-interval 5 --power-cap 8000 --control-interval 2
```

Yêu cầu: đã bật **per-job DVFS + reschedule**. Nếu chưa, hành vi sẽ gần `cap_uniform`. (Thuật toán bám đúng aggregation từ khung trong paper.

## `joint_nf` (tối ưu đồng thời số GPU & tần số theo SLO)

Mỗi khi job sẵn sàng chạy, duyệt lưới $n \in [1 .. N_{\max}], f \in \text{freq\_levels}$; chọn nghiệm **min năng lượng** (hoặc carbon/cost) **thoả SLA**.

```bash
python run_sim_paper.py --algo joint_nf \
  --duration 1200 --log-interval 5
```

Gợi ý:

* Gán `job.deadline` cho inference nếu muốn ràng buộc SLO.
* Hệ số $T(n,f)$ phải hợp lý.
  Cơ sở lựa chọn: DVFS-aware, SLO-aware scaling trên LLM cho thấy tồn tại điểm tốt (vd. \~1050 MHz trên A100) đem lại +\~37% hiệu quả năng lượng với ảnh hưởng hiệu năng nhỏ.

## `bandit` (UCB1)

Mỗi DC/job_type xem mỗi $f$ là một arm. Khi job tới: **chọn** $f$ theo UCB1; khi job xong: **cập nhật** reward = −(năng lượng/đơn vị). Không cần mô hình chính xác, tự thích nghi theo workload.

```bash
python run_sim_paper.py --algo bandit \
  --duration 1200 --log-interval 5
```

Gợi ý:

* Đặt `init_explore=1` (mặc định trong learner) để thử qua mọi $f$.
* Reward có thể chuyển sang **carbon** (−E·CI) hoặc **cost** (−E·price).
  Lý thuyết UCB1 và biến thể (UCB1-Tuned) là nền tảng kinh điển cho tối ưu thăm-dò/khai-thác.

## `carbon_cost` (carbon-/cost-aware)

Chọn $n,f$ sao cho **min** $E\cdot\text{CI(dc)}$ **hoặc** $(E/3.6\text{e}6)\cdot\text{price(dc, hour)}$.
(Ở tầng routing có thể đi theo DC có CI thấp — Carbon-Intelligent Compute.)

```bash
python run_sim_paper.py --algo carbon_cost \
  --duration 1200 --log-interval 5
```

## `debug` Cố định $n,f$
Cố định $n,f$ cho các job, hiện dùng để kiểm tra hành vi DC chỉ training jobs: 
* `--num_fixed_gpus` (default = `1`): Số GPUs cố định gán cho 1 job.
* `--fixed_freq` (default = `None`): Tần số GPU cố định gán cho 1 job.
  * Mặc định `None` sẽ chọn `best_energy_freq` với $n$ xác định.
  * `best_energy_freq` **có thể thay đổi theo từng job**.

```bash
python run_sim_paper.py --algo debug --num_fixed_gpus 1 \
--inf-mode off --trn-mode poisson --trn-rate 0.02 --duration 86400 --log-interval 10
```

## Our algorithm

* `--elastic-scaling` (default = `False`): hiện chỉ dùng cho **RL** - Tác nhân sẽ preempt và phân bổ lại tài nguyên cho training jobs sau mỗi sự kiện training job completion. 
  * Qua thực nghiệm thấy (implement) chưa hiệu quả, làm tăng hàng đợi nhiều.

```bash
python run_sim_paper.py --algo eco_route --eco-objective energy --duration 1200 --log-interval 5
```

```bash
python run_sim_paper.py --algo rl_energy --rl-alpha 0.1 --rl-gamma 0.0 --rl-eps 0.2 --rl-eps-decay 0.995 --rl-eps-min 0.02 --rl-n-cand 2 --duration 1200 --log-interval 5
```

```bash
python run_sim_paper.py --algo rl_energy_adv --rl-alpha 0.1 --rl-gamma 0.1 --rl-eps 0.05 --rl-eps-decay 0.999 --rl-eps-min 0.01 --rl-tau 0.1 --rl-clip-grad 5.0 --rl-baseline-beta 0.01 --rl-n-cand 2
```

## Vẽ đồ thị

Mỗi run (một cấu hình/thuật toán) để trong một thư mục có:

* `cluster_log.csv`
* `job_log.csv`

Chạy:

```bash
python plot_sim_result.py --run baseline=./runs/baseline --run cap_greedy=./runs/cap_greedy --run carbon=./runs/carbon_cost --outdir ./figs --bin 5
```

* `NAME=DIR`: tên muốn hiển thị trên legend và thư mục chứa CSV. Ví dụ: baseline=./runs/baseline
* `--outdir`: nơi lưu hình.
* `--bin`: kích thước bin (giây) cho biểu đồ throughput.


### Log_path của mô phỏng
- Khi chạy mô phỏng, mặc định tạo thư mục con **cùng tên thuật toán** để lưu log ở trong thư mục cha.
  - Nếu **truyền cả thư mục con vào** thì sẽ lưu ở thư mục con đó.
- Ví dụ: 
```bash
python run_sim_paper.py --algo debug --log-path results  # lưu log ở `results/debug/`
```
```bash
python run_sim_paper.py --algo debug --log-path results/debug_1  # lưu log ở `results/debug_1/`
```


## Batch Script: `run_all.bat`

- Tự động chạy mô phỏng cho nhiều cấu hình và plot **tất cả các runs** trong folder.
  - Hiện file mẫu chạy `debug` với số GPUs chạy từ $1$ đến $8$, $f$ không truyền (tức `None`).
- Các tham số phần CONFIG SECTION: 
  - `SKIP_SIM`:
    - `0`: chạy cả mô phỏng và vẽ hình.
    - `1`: bỏ qua mô phỏng, chỉ vẽ hình từ tất cả các runs (tất cả các folder) hiện có.
      - Tức, nếu chạy `SKIP_SIM = 1`, cần đảm bảo tất cả các folder trong folder cha đều là simulation runs. 
  - Các tham số còn lại đều của mô phỏng chạy và vẽ đồ thị.
---