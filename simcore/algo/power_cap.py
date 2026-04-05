from simcore.algo.freq_load_agg import TaskState, aggregate_with_atoms


def cap_uniform(sim, deficit: float) -> None:
    """Giảm f theo bước rời rạc, mỗi bước chọn DC có ∆P lớn nhất."""
    iter_guard = 10000
    while deficit > 1e-6 and iter_guard > 0:
        iter_guard -= 1
        best_dc = None
        best_dp = 0.0
        best_next_f = None

        for dc in sim.dcs.values():
            levels = dc.freq_levels
            if not levels:
                continue

            # snap index
            try:
                idx = levels.index(dc.current_freq)
            except ValueError:
                idx = min(range(len(levels)), key=lambda i: abs(levels[i] - dc.current_freq))

            if idx == 0:
                continue

            f_now = levels[idx]
            f_lo = levels[idx - 1]

            p_now = sim._estimate_dc_power(dc, f_now)
            p_lo = sim._estimate_dc_power(dc, f_lo)
            dp = p_now - p_lo

            if dp > best_dp + 1e-9:
                best_dc = dc
                best_dp = dp
                best_next_f = f_lo

        if best_dc is None or best_dp <= 1e-9:
            break

        best_dc.current_freq = best_next_f
        deficit -= best_dp


def run_power_cap_control(sim) -> None:
    """
    Power-cap controller:
      - cap_uniform: hạ đồng loạt theo DC, mỗi bước chọn DC có ∆P lớn nhất.
      - cap_greedy : hạ theo 'atoms' per-job (từ aggregate), tái-ước-lượng sau mỗi bước.
    Có hysteresis nhỏ để tránh rung quanh power_cap.
    """
    if sim.power_cap <= 0:
        return

    algo = getattr(sim, "algo", "default_policy")

    if algo not in ("cap_uniform", "cap_greedy"):
        # (tuỳ chọn) vài heuristic nhẹ cho các algo khác
        if algo in ("eco_route", "carbon_cost"):
            # hạ f về min khi DC rảnh để giảm idle power
            for dc in sim.dcs.values():
                if dc.busy_gpus == 0 and dc.freq_levels:
                    dc.current_freq = min(dc.freq_levels)
        return

    # 1) Ước lượng tổng P hiện tại
    totalP = 0.0
    for dc in sim.dcs.values():
        f_now = getattr(dc, "current_freq", 1.0)
        totalP += sim._estimate_dc_power(dc, f_now)

    # Hysteresis: nếu đã dưới cap một chút thì thôi
    cap_margin = getattr(sim, "cap_margin", 5.0)
    if totalP <= sim.power_cap - cap_margin:
        return

    # Thâm hụt cần cắt
    deficit = max(0.0, totalP - sim.power_cap)
    if deficit <= 1e-6:
        return

    # 2) cap_uniform
    if algo == "cap_uniform":
        cap_uniform(sim, deficit)
        return

    # 3) cap_greedy
    iter_guard = 10000
    while deficit > 1e-6 and iter_guard > 0:
        iter_guard -= 1
        # 3.1 Gom tasks (bỏ qua DC không có freq_levels hoặc job đã ở f min)
        tasks = []
        for dc in sim.dcs.values():
            levels = getattr(dc, "freq_levels", None) or []
            if not levels:
                continue

            f_min = min(levels)
            for job, g in list(dc.running_jobs.values()):
                pC, tC = sim.coeffs_map[(dc.name, job.jtype)]
                cur_f = getattr(job, "f_used", 0.0) or dc.current_freq

                if cur_f <= f_min + 1e-12:
                    continue

                tasks.append(
                    TaskState(
                        job_id=job.jid,
                        dc_name=dc.name,
                        n=g,
                        f=cur_f,
                        freq_levels=levels,
                        p_coeffs=pC,
                        t_coeffs=tC,
                    )
                )

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

            dc = sim.dcs.get(atom.dc_name)
            if dc is None:
                continue

            tup = dc.running_jobs.get(atom.job_id)
            if not tup:
                continue

            job, g = tup
            cur_f = getattr(job, "f_used", dc.current_freq)

            # bỏ qua atoms không đi xuống
            if atom.f_to >= cur_f - 1e-12:
                continue
            # Thực hiện reschedule về f_to
            sim._reschedule_job(dc, job, g, atom.f_to)
            applied_any = True

            # 3.4 Tái-ước-lượng totalP CHÍNH XÁC sau khi đổi f
            new_totalP = 0.0
            for _dc in sim.dcs.values():
                new_totalP += sim._estimate_dc_power(_dc, getattr(_dc, "current_freq", 1.0))

            totalP = new_totalP
            deficit = max(0.0, totalP - sim.power_cap)

            if deficit <= 1e-6:
                break
        # Nếu vòng này không áp dụng được atom nào → thoát
        if not applied_any:
            break