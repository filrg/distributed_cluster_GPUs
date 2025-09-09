# Cách đọc các giá trị trong file log

## `cluster_log.csv`

### Các cột & đơn vị

* **time\_s** — thời gian mô phỏng (giây) tại mốc log.
* **dc** — tên data center.
* **freq** — mức tần số *đặt ở cấp DC* (`dc.current_freq`, thường chuẩn hoá 0–1).
  *Lưu ý:* nếu bật **per-job DVFS**, mỗi job có thể chạy ở `job.f_used` ≠ `freq`. Xem chi tiết `f_used` trong `job_log.csv`.
* **busy** — số GPU đang bận (đang chạy job) tại DC.
* **free** — số GPU rảnh = `total_gpus − busy`.
* **run\_total** — tổng số job đang chạy tại DC.
* **run\_inf** — số job inference đang chạy.
* **run\_train** — số job training đang chạy.
* **q\_inf** — chiều dài hàng đợi job inference tại DC.
* **q\_train** — chiều dài hàng đợi job training tại DC.
* **util\_inst** — GPU occupancy tức thời = `busy / total_gpus` (0..1).
* **util\_avg** — GPU occupancy trung bình theo thời gian từ đầu mô phỏng đến `time_s`:

$$
\textstyle util\_avg = \frac{\sum \text{busy}(t)\cdot \Delta t}{\text{total\_gpus}\cdot (time\_s - t_{start})}
$$
* **power\_W** — **công suất tức thời** của DC (W), tính theo mô hình paper:

$$
\textstyle P_{\text{DC}}(t) = \sum_{\text{job }j} n_j \cdot P_{\text{gpu}}(f_j)\;+\;(\text{free})\cdot
\begin{cases}
p_{\text{sleep}}, & \text{nếu bật power\_gating}\\
p_{\text{idle}}, & \text{nếu không}
\end{cases}
$$

với $f_j = job.f\_used$ (nếu có) hoặc `freq` của DC.
* **energy\_kJ** — **năng lượng luỹ kế** của DC từ đầu mô phỏng đến `time_s` (kJ):

$$
\textstyle E(t) = \int\_0^{t} P_{\text{DC}}(\tau)\,d\tau \quad (\text{đổi J→kJ})
$$

## `job_log.csv`

### Cột & ý nghĩa

* **jid** — mã job.
* **ingress** — ingress node nơi yêu cầu xuất phát.
* **type** — loại job: `inference` | `training`.
* **size** — số “đơn vị công việc” (unit) của job. Tổng thời gian tính theo `size`.
* **dc** — data center xử lý job.
* **f\_used** — tần số dùng cho job (chuẩn hoá 0–1). Nếu bật per-job DVFS thì đây là `job.f_used`; nếu không, bằng `dc.current_freq` lúc start.
* **n\_gpus** — số GPU cấp cho job (chạy song song).
* **net\_lat\_s** (s) — độ trễ mạng trước khi job “vào máy” (transfer).
* **start\_s**, **finish\_s** (s) — mốc thời gian mô phỏng khi job bắt đầu/kết thúc compute.
* **latency\_s** (s) — latency compute thực tế = `finish_s − start_s` (không gồm mạng, không gồm chờ hàng đợi).
* **T\_pred** (s/unit) — thời gian dự đoán trên **mỗi unit**: $T(n,f)$ từ `TrainLatencyCoeffs`.
* **P\_pred** (W) — công suất dự đoán của **job** khi chạy: $P(n,f)=n\cdot P_{\text{gpu}}(f)$ từ `TrainPowerCoeffs`.
* **E\_pred** (J/unit) — năng lượng dự đoán **mỗi unit** của job: $E_{\text{unit}}=P(n,f)\cdot T(n,f)$.
