import numpy as np
from typing import Dict
from simcore.rl.replay import save_offline_npz


def build_offline_npz_from_logs(logs, path: str):
    """Ví dụ chuyển log mô phỏng thành dataset offline.
    `logs` là iterable của bản ghi env: (s, a_dict, r, costs_dict, s_next, done, mask_dc, mask_g)
    """
    states, next_states = [], []
    a_dc, a_g = [], []
    rewards, dones = [], []
    mask_dc_list, mask_g_list = [], []
    # costs tách cột theo tên
    cost_names = None
    cost_buf: Dict[str, list] = {}

    for (s, a, r, costs, s2, d, mdc, mg) in logs:
        states.append(s)
        next_states.append(s2)
        a_dc.append(a['dc'])
        a_g.append(a['g'])
        rewards.append(r)
        dones.append(d)
        mask_dc_list.append(mdc)
        mask_g_list.append(mg)
        if cost_names is None:
            cost_names = list(costs.keys())
            for k in cost_names:
                cost_buf[k] = []
        for k in cost_names:
            cost_buf[k].append(costs[k])

    data = {
        's': np.asarray(states, dtype=np.float32),
        's_next': np.asarray(next_states, dtype=np.float32),
        'a_dc': np.asarray(a_dc, dtype=np.int64),
        'a_g': np.asarray(a_g, dtype=np.int64),
        'r': np.asarray(rewards, dtype=np.float32),
        'done': np.asarray(dones, dtype=np.float32),
        'mask_dc': np.asarray(mask_dc_list, dtype=np.bool_),
        'mask_g': np.asarray(mask_g_list, dtype=np.bool_),
    }
    for k, v in cost_buf.items():
        data[f'costs/{k}'] = np.asarray(v, dtype=np.float32)
    save_offline_npz(path, data)
