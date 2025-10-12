import torch

from simcore.rl.replay import ReplayBuffer, Transition


class HybridSACTrainer:
    """Vòng lặp train tối giản (demo) — thay bằng trainer của bạn nếu đã có.
    """
    def __init__(self, agent, buffer: ReplayBuffer, gamma=0.99, batch_size=256):
        self.agent = agent
        self.buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size

    def step_env_and_learn(self, env):
        # 1) Lấy obs & mask từ env
        obs = env.get_obs_vector()  # np.ndarray
        mask_dc, mask_g = env.get_action_masks()  # np arrays [N_dc], [N_g]
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        mask_dc_t = torch.from_numpy(mask_dc).unsqueeze(0).bool()
        mask_g_t = torch.from_numpy(mask_g).unsqueeze(0).bool()
        # 2) Chọn hành động
        a = self.agent.select_action(obs_t, mask_dc_t, mask_g_t)
        # 3) Thực thi
        (next_obs, r, done, info) = env.step(a)
        costs = info.get('costs', {})
        tr = Transition(s=obs, s_next=next_obs, a_dc=a['dc'], a_g=a['g'], a_f=a['f'], r=r, costs=costs, done=done,
                        mask_dc=mask_dc, mask_g=mask_g)
        self.buffer.add(tr)
        # 4) Học khi đủ dữ liệu
        if self.buffer.size >= self.batch_size:
            batch = self.buffer.sample(self.batch_size)
            stats = self.agent.train_step(batch)
            return stats
        return {}
    