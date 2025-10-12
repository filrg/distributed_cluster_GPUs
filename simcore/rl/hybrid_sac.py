from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import FreqBounds, SquashedNormal, sample_categorical


class HybridActor(nn.Module):
    """Chính sách Hybrid: chọn DC (categorical), GPU count (categorical), DVFS freq (squashed Gaussian).
    - Input: latent h (từ encoder)
    - Output: logits_dc [B, N_dc], logits_g [B, N_g], (mu_f, log_std_f) [B,1]
    """
    def __init__(self, latent_dim: int, n_dc: int, n_gpus_choices: int, freq_bounds: FreqBounds):
        super().__init__()
        self.n_dc = n_dc
        self.n_g = n_gpus_choices
        self.freq_bounds = freq_bounds
        hid = 256
        self.head_dc = nn.Sequential(
            nn.Linear(latent_dim, hid), nn.ReLU(), nn.Linear(hid, n_dc)
        )
        self.head_g = nn.Sequential(
            nn.Linear(latent_dim, hid), nn.ReLU(), nn.Linear(hid, n_gpus_choices)
        )
        self.head_f_mu = nn.Sequential(
            nn.Linear(latent_dim, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.head_f_logstd = nn.Sequential(
            nn.Linear(latent_dim, hid), nn.ReLU(), nn.Linear(hid, 1)
        )

    def forward(self, h: torch.Tensor):
        logits_dc = self.head_dc(h)
        logits_g = self.head_g(h)
        mu_f = self.head_f_mu(h)
        log_std_f = self.head_f_logstd(h)
        return logits_dc, logits_g, mu_f, log_std_f

    def sample(self, h: torch.Tensor, mask_dc: Optional[torch.Tensor], mask_g: Optional[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        logits_dc, logits_g, mu_f, log_std_f = self.forward(h)
        # categorical parts
        a_dc, logp_dc = sample_categorical(logits_dc, mask_dc)
        a_g, logp_g = sample_categorical(logits_g, mask_g)
        # continuous part
        f_dist = SquashedNormal(mu_f, log_std_f, self.freq_bounds.f_min, self.freq_bounds.f_max)
        a_f, logp_f = f_dist.rsample()
        # tổng log_prob
        logp = logp_dc + logp_g + logp_f
        return {"dc": a_dc, "g": a_g, "f": a_f.squeeze(-1)}, logp

    def deterministic(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits_dc, logits_g, mu_f, log_std_f = self.forward(h)
        dc = torch.argmax(logits_dc, dim=-1)
        g = torch.argmax(logits_g, dim=-1)
        f = SquashedNormal(mu_f, log_std_f, self.freq_bounds.f_min, self.freq_bounds.f_max).mode()
        return {"dc": dc, "g": g, "f": f.squeeze(-1)}


class QuantileCritic(nn.Module):
    """Quantile critic đôi (twin) cho SAC.
    - Input: latent state h + action embedding (one-hot dc, one-hot g, norm f)
    - Output: 2 bộ quantiles: [B, N_quantiles]
    """
    def __init__(self, latent_dim: int, n_dc: int, n_g: int, n_quantiles: int = 32):
        super().__init__()
        self.n_dc, self.n_g, self.nq = n_dc, n_g, n_quantiles
        a_dim = n_dc + n_g + 1
        hid = 256
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_quantiles)
        )
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_quantiles)
        )

    def forward(self, h: torch.Tensor, a_dc: torch.Tensor, a_g: torch.Tensor, a_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # one-hot encode dc, g; normalize f roughly to [0,1] by min/max in batch if available → conservative: tanh of scaled
        B = h.size(0)
        dc_oh = F.one_hot(a_dc, num_classes=self.n_dc).float()
        g_oh = F.one_hot(a_g, num_classes=self.n_g).float()
        a_f = a_f.view(B, 1)
        inp = torch.cat([h, dc_oh, g_oh, a_f], dim=-1)
        q1 = self.q1(inp)
        q2 = self.q2(inp)
        return q1, q2


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """QR-DQN style quantile Huber loss.
    pred: [B, N] quantiles, target: [B, N] (stopgrad), taus: [N]
    """
    # pairwise delta (broadcast)
    delta = target.unsqueeze(2) - pred.unsqueeze(1)  # [B, N_tgt, N_pred]
    abs_delta = torch.abs(delta)
    huber = torch.where(abs_delta <= 1.0, 0.5 * delta ** 2, 1.0 * (abs_delta - 0.5))
    # quantile weights
    tau = taus.view(1, -1, 1)  # [1, N_tgt, 1]
    weight = torch.abs((delta.detach() < 0).float() - tau)
    loss = (weight * huber).mean()
    return loss


class HybridSAC(nn.Module):
    """Gói actor-critic + update step cho Hybrid SAC phân phối (quantile critic).
    Chỉ là khung, bạn có thể nhúng vào vòng lặp train của `RLEnergyAgentAdv`.
    """
    def __init__(self, encoder: nn.Module, actor: HybridActor, critic: QuantileCritic,
                 n_quantiles: int = 32, alpha: float = 0.2, actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.target_critic = QuantileCritic(encoder.net[-2].out_features if hasattr(encoder, 'net') else 256,
                                            actor.n_dc, actor.n_g, n_quantiles)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.nq = n_quantiles
        self.taus = torch.linspace(1.0/(2*self.nq), 1 - 1.0/(2*self.nq), self.nq)
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
        self.actor_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.actor.parameters()) + [self.log_alpha], lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device or torch.device('cpu')
        self.to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, state: torch.Tensor, mask_dc: Optional[torch.Tensor], mask_g: Optional[torch.Tensor], deterministic=False):
        state = state.to(self.device)
        h = self.encoder(state)
        if deterministic:
            return self.actor.deterministic(h)
        a, logp = self.actor.sample(h, mask_dc, mask_g)
        a["logp"] = logp
        return a

    @torch.no_grad()
    def _target_quantiles(self, h_next: torch.Tensor, mask_dc_n: Optional[torch.Tensor], mask_g_n: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample next action from current actor
        logits_dc, logits_g, mu_f, log_std_f = self.actor(h_next)
        # re-sample for entropy term
        a_dc_n, logp_dc_n = sample_categorical(logits_dc, mask_dc_n)
        a_g_n, logp_g_n = sample_categorical(logits_g, mask_g_n)
        f_dist = SquashedNormal(mu_f, log_std_f, self.actor.freq_bounds.f_min, self.actor.freq_bounds.f_max)
        a_f_n, logp_f_n = f_dist.rsample()
        logp_n = logp_dc_n + logp_g_n + logp_f_n
        q1_t, q2_t = self.target_critic(h_next, a_dc_n, a_g_n, a_f_n.squeeze(-1))
        q_min = torch.min(q1_t, q2_t)  # [B, Nq]
        # entropy temperature term subtract
        q_min = q_min - self.alpha.detach().view(1,)*logp_n.view(-1,1)
        return q_min, logp_n

    def update(self, batch: Dict[str, torch.Tensor], gamma: float = 0.99, tau: float = 0.005, target_entropy: float = -3.0):
        """batch keys: s, a_dc, a_g, a_f, r_eff, s_next, done, mask_dc, mask_g, mask_dc_n, mask_g_n
        r_eff đã bao gồm Lagrangian (Xem CMDP wrapper).
        """
        s = batch['s'].to(self.device)
        s_next = batch['s_next'].to(self.device)
        a_dc = batch['a_dc'].to(self.device)
        a_g = batch['a_g'].to(self.device)
        a_f = batch['a_f'].to(self.device)
        r_eff = batch['r_eff'].to(self.device).unsqueeze(-1)
        done = batch['done'].to(self.device).unsqueeze(-1)
        mask_dc = batch.get('mask_dc', None)
        mask_g = batch.get('mask_g', None)
        mask_dc_n = batch.get('mask_dc_n', mask_dc)
        mask_g_n = batch.get('mask_g_n', mask_g)

        # ----- Critic update -----
        h = self.encoder(s)
        h_next = self.encoder(s_next).detach()
        with torch.no_grad():
            q_next, logp_n = self._target_quantiles(h_next, mask_dc_n, mask_g_n)  # [B,Nq]
            target = r_eff + (1 - done) * gamma * q_next  # broadcast to [B,Nq]
            target = target.detach()
        q1, q2 = self.critic(h, a_dc, a_g, a_f)
        taus = self.taus.to(self.device)
        loss_q1 = quantile_huber_loss(q1, target, taus)
        loss_q2 = quantile_huber_loss(q2, target, taus)
        loss_critic = loss_q1 + loss_q2
        self.critic_opt.zero_grad(set_to_none=True)
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # ----- Actor & temperature update -----
        h_detached = self.encoder(s)  # share encoder; allow actor loss gradients flow into encoder
        logits_dc, logits_g, mu_f, log_std_f = self.actor(h_detached)
        # sample actions for policy gradient
        a_dc_pi, logp_dc = sample_categorical(logits_dc, mask_dc)
        a_g_pi, logp_g = sample_categorical(logits_g, mask_g)
        f_dist = SquashedNormal(mu_f, log_std_f, self.actor.freq_bounds.f_min, self.actor.freq_bounds.f_max)
        a_f_pi, logp_f = f_dist.rsample()
        logp = logp_dc + logp_g + logp_f
        # Q under current critic
        q1_pi, q2_pi = self.critic(h_detached, a_dc_pi, a_g_pi, a_f_pi.squeeze(-1))
        q_pi = torch.min(q1_pi, q2_pi).mean(dim=-1)  # mean over quantiles → [B]
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        # temperature loss (automatic entropy tuning)
        temp_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        (actor_loss + temp_loss).backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.actor.parameters()), 5.0)
        self.actor_opt.step()

        # ----- Soft update targets -----
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        return {
            'loss_critic': loss_critic.item(),
            'loss_actor': actor_loss.item(),
            'loss_temp': temp_loss.item(),
            'alpha': self.alpha.item(),
        }
