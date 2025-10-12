from typing import Dict, Any, Optional
import torch

from simcore.rl.encoders import MLPStateEncoder
from simcore.rl.hybrid_sac import HybridActor, QuantileCritic, HybridSAC
from simcore.rl.utils import FreqBounds
from simcore.rl.cmdp_wrapper import LagrangianCMDP, ConstraintSpec


class RLEnergyAgentAdvUpgr:
    """Khung agent nâng cấp hoà nhập với môi trường hiện tại.
    Giả định env cung cấp:
        - obs vector hoá (đã concat preference nếu cần)
        - mask_dc [N_dc] và mask_g [N_g] cho bước hiện tại
        - step(action_dict) nhận {dc:int, g:int, f:float}
        - reward r và cost_dict (ví dụ: {"latency_p99":..., "power":...})
    """
    def __init__(self, obs_dim: int, n_dc: int, n_g_choices: int, freq_minmax: tuple,
                 constraints: Dict[str, float], device: str = 'cpu'):
        self.device = torch.device(device)
        self.encoder = MLPStateEncoder(obs_dim).to(self.device)
        self.actor = HybridActor(256, n_dc, n_g_choices, FreqBounds(*freq_minmax)).to(self.device)
        self.critic = QuantileCritic(256, n_dc, n_g_choices).to(self.device)
        self.algo = HybridSAC(self.encoder, self.actor, self.critic, device=self.device)
        specs = {k: ConstraintSpec(name=k, target=v) for k, v in constraints.items()}
        self.cmdp = LagrangianCMDP(specs)

    def select_action(self, obs: torch.Tensor, mask_dc: Optional[torch.Tensor], mask_g: Optional[torch.Tensor], deterministic=False) -> Dict[str, Any]:
        a = self.algo.act(obs, mask_dc, mask_g, deterministic=deterministic)
        return {"dc": int(a["dc"].item()), "g": int(a["g"].item()), "f": float(a["f"].item())}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Tính r_eff từ CMDP
        r = batch['r']
        costs = batch['costs']  # dict name->tensor [B]
        r_eff = r.clone()
        for k, v in costs.items():
            target = self.cmdp.constraints[k].target
            e = (v - target).clamp(min=0.0)
            r_eff = r_eff - self.cmdp.lmbda[k].to(r.device) * e
        batch = {**batch, 'r_eff': r_eff}
        stats = self.algo.update(batch)
        # cập nhật λ theo trung bình cost
        lam_stats = self.cmdp.update_lagrange(costs)
        stats.update(lam_stats)
        return stats
