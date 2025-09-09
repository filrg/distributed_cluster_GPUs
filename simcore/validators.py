from typing import Iterable, List
from .models import GPUType


def validate_gpus(gpus: Iterable[GPUType], strict: bool = False) -> List[str]:
    """
    Trả về danh sách cảnh báo. Nếu strict=True và có cảnh báo, raise ValueError.
    Kiểm tra chính:
      - p_idle + p_peak > tdp  (vượt TDP)
      - p_sleep > p_idle
      - giá trị âm / alpha ngoài biên hợp lý
    """
    msgs: List[str] = []
    seen = set()  # tránh lặp nếu nhiều DC dùng cùng loại GPU

    for g in gpus:
        if id(g) in seen:
            continue
        seen.add(id(g))
        prefix = f"[GPUType:{g.name}]"

        # 1) Sanity số học
        if g.p_idle < 0 or g.p_peak < 0 or g.p_sleep < 0:
            msgs.append(f"{prefix} Giá trị âm (p_idle={g.p_idle}, p_peak={g.p_peak}, p_sleep={g.p_sleep}).")

        # 2) Sleep không thể lớn hơn idle (thường là nhỏ hơn hoặc xấp xỉ)
        if g.p_sleep > g.p_idle + 1e-6:
            msgs.append(f"{prefix} p_sleep ({g.p_sleep} W) > p_idle ({g.p_idle} W). Kiểm tra lại cấu hình/đo đạc.")

        # 3) alpha nên nằm trong vùng hợp lý (chỉ để nhắc nhở)
        if not (1.0 <= g.alpha <= 5.0):
            msgs.append(f"{prefix} alpha={g.alpha} ngoài biên [1,5]; nên fit từ dữ liệu đo.")

        # 4) Kiểm tra vượt TDP nếu có tdp
        if g.tdp is not None:
            total = g.p_idle + g.p_peak
            if total > g.tdp + 1e-6:
                msgs.append(f"{prefix} p_idle + p_peak = {total:.1f} W > TDP {g.tdp:.1f} W. "
                            f"Đặt p_peak ≈ (TDP - p_idle) nếu dùng baseline.")
            # (tuỳ chọn) cảnh báo nếu quá thấp so với TDP -> có thể nhập nhầm đơn vị
            if total < 0.5 * g.tdp:
                msgs.append(f"{prefix} p_idle + p_peak = {total:.1f} W << TDP {g.tdp:.1f} W (<=50%).")

    if strict and msgs:
        raise ValueError("GPU config validation failed:\n" + "\n".join(msgs))
    return msgs
