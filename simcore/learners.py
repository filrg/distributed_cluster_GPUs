from collections import defaultdict
import math, random

class BanditDVFS:
    """
    UCB1 trên tập tần số rời rạc per (dc, job_type).
    reward = - cost_per_unit (ở đây cost = energy_per_unit, hoặc carbon_cost_per_unit).
    """
    def __init__(self, init_explore: int = 1, objective: str = "energy"):
        self.N = defaultdict(int)       # số lần chọn arm
        self.S = defaultdict(float)     # tổng reward
        self.t = 0
        self.objective = objective
        self.init_explore = init_explore

    def _key(self, dc_name, job_type, f):
        return (dc_name, job_type, float(f))

    def select(self, dc_name, job_type, freq_levels):
        self.t += 1
        # explore mỗi arm ít nhất init_explore lần
        for f in freq_levels:
            k = self._key(dc_name, job_type, f)
            if self.N[k] < self.init_explore:
                return f
        # UCB1
        best_f, best_ucb = None, -1e9
        for f in freq_levels:
            k = self._key(dc_name, job_type, f)
            n = self.N[k]
            mean = self.S[k]/n if n>0 else 0.0
            ucb = mean + math.sqrt(2.0*math.log(self.t)/n)
            if ucb > best_ucb:
                best_ucb, best_f = ucb, f
        return best_f

    def update(self, dc_name, job_type, f, cost_per_unit):
        # reward = -cost
        k = self._key(dc_name, job_type, float(f))
        self.N[k] += 1
        self.S[k] += -float(cost_per_unit)
