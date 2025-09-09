import math, random
from dataclasses import dataclass


def expovariate_safe(lmbda: float) -> float:
    return float('inf') if lmbda <= 0 else random.expovariate(lmbda)


@dataclass
class ArrivalConfig:
    mode: str  # 'poisson' | 'sinusoid'
    rate: float
    amp: float = 0.0
    period: float = 3600.0

    def lambda_t(self, t: float) -> float:
        if self.mode == 'poisson':
            return self.rate
        elif self.mode == 'sinusoid':
            return max(0.0, self.rate * (1.0 + self.amp *
                                         math.sin(2 * math.pi * (t % self.period) / self.period)))
        raise ValueError("Unknown mode")

    def next_interarrival(self, t: float) -> float:
        if self.mode == 'poisson':
            return expovariate_safe(self.rate)
        elif self.mode == 'sinusoid':
            # thinning cho Poisson tốc độ thay đổi
            max_rate = self.rate * (1.0 + abs(self.amp))
            while True:
                w = expovariate_safe(max_rate)
                t_candidate = t + w
                if random.random() <= self.lambda_t(t_candidate) / max_rate:
                    return w
        raise ValueError("Unknown mode")
