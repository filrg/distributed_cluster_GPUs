import math
from typing import Dict, Tuple

class InferenceLUT:
    """LUT for 1-GPU inference: (f,b) -> (t1, e1). Fill from offline profiling."""
    def __init__(self, t1_table: Dict[Tuple[float,int], float],
                       e1_table: Dict[Tuple[float,int], float]):
        self.t1 = dict(t1_table)
        self.e1 = dict(e1_table)

    def time_and_energy(self, n: int, f: float, b: int, l: int):
        key = (float(f), int(b))
        if key not in self.t1 or key not in self.e1:
            raise KeyError(f"No LUT entry for f={f}, b={b}")
        t1 = self.t1[key]
        e1 = self.e1[key]
        batches = math.ceil(max(0, int(l)) / max(1, int(n*b)))
        T_total = batches * t1
        E_total = batches * n * e1
        return T_total, E_total
