from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Transition:
    # state & next
    s: np.ndarray
    s_next: np.ndarray
    # actions (dc index, gpu choice index, freq float)
    a_dc: int
    a_g: int
    a_f: float
    # raw reward and costs (dictionary keys must align with ConstraintSpec names)
    r: float
    costs: Dict[str, float]
    done: bool
    # optional preference vector used when training preference-conditioned policy
    pref: Optional[np.ndarray] = None
    # optional masks for feasibility
    mask_dc: Optional[np.ndarray] = None
    mask_g: Optional[np.ndarray] = None


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.storage: List[Transition] = [None] * capacity

    def add(self, tr: Transition):
        self.storage[self.ptr] = tr
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = [self.storage[i] for i in idxs]
        # build tensors
        s = torch.from_numpy(np.stack([b.s for b in batch])).float().to(self.device)
        s_next = torch.from_numpy(np.stack([b.s_next for b in batch])).float().to(self.device)
        a_dc = torch.tensor([b.a_dc for b in batch], dtype=torch.long, device=self.device)
        a_g = torch.tensor([b.a_g for b in batch], dtype=torch.long, device=self.device)
        a_f = torch.tensor([b.a_f for b in batch], dtype=torch.float32, device=self.device)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        # pack costs into dict of tensors
        # assume all have same keys
        cost_keys = list(batch[0].costs.keys()) if batch[0].costs else []
        cost_tensors = {k: torch.tensor([b.costs[k] for b in batch], dtype=torch.float32, device=self.device) for k in cost_keys}
        # masks (optional)
        mask_dc = None
        mask_g = None
        if batch[0].mask_dc is not None:
            mask_dc = torch.from_numpy(np.stack([b.mask_dc for b in batch])).bool().to(self.device)
        if batch[0].mask_g is not None:
            mask_g = torch.from_numpy(np.stack([b.mask_g for b in batch])).bool().to(self.device)
        return {
            's': s, 's_next': s_next,
            'a_dc': a_dc, 'a_g': a_g, 'a_f': a_f,
            'r': r, 'done': done,
            'costs': cost_tensors,
            'mask_dc': mask_dc, 'mask_g': mask_g,
        }

# ---- Offline dataset (schema + loader) ----
# Lưu ở dạng npz (nhẹ, nhanh) với các mảng song song cùng chiều N
# Keys bắt buộc: s, s_next, a_dc, a_g, a_f, r, done
# Keys tuỳ chọn: costs/<name>, pref, mask_dc, mask_g

def save_offline_npz(path: str, data: Dict[str, np.ndarray]):
    np.savez_compressed(path, **data)

def load_offline_npz(path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        v = z[k]
        if k.startswith('costs/'):
            # costs/<name> sẽ được gom lại sau
            out.setdefault('costs', {})
            out['costs'][k.split('/', 1)[1]] = torch.tensor(v)
        else:
            out[k] = torch.tensor(v)
    # chuyển device + dtype
    td = {}
    for k, v in out.items():
        if k == 'costs':
            td[k] = {ck: cv.float().to(device) for ck, cv in v.items()}
        else:
            td[k] = v.to(device)
    return td
