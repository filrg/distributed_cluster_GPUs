import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Ingress:
    """Biểu diễn cổng vào (POP/edge gateway)."""
    name: str
    region: str

@dataclass
class Edge:
    to: str
    latency_ms: float
    capacity_gbps: float = math.inf
    cost_per_GB: float = 0.0  # egress cost cộng dồn theo path

class Graph:
    """Đồ thị có hướng cho WAN; node = tên (Ingress/DC)."""
    def __init__(self):
        self.adj: Dict[str, List[Edge]] = {}

    def add_edge(self, u: str, v: str, latency_ms: float,
                 capacity_gbps: float = math.inf, cost_per_GB: float = 0.0):
        self.adj.setdefault(u, []).append(Edge(v, latency_ms, capacity_gbps, cost_per_GB))

    def shortest_path_latency(self, src: str, dst: str) -> Tuple[float, List[str], float, float]:
        """Trả về (latency_s, path_nodes, bottleneck_Gbps, sum_cost_per_GB). Dijkstra theo latency."""
        dist: Dict[str, float] = {src: 0.0}
        prev: Dict[str, Tuple[str, Edge]] = {}
        pq: List[Tuple[float, str]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst: break
            if d > dist.get(u, math.inf): continue
            for e in self.adj.get(u, []):
                nd = d + e.latency_ms
                if nd < dist.get(e.to, math.inf):
                    dist[e.to] = nd
                    prev[e.to] = (u, e)
                    heapq.heappush(pq, (nd, e.to))
        if dst not in dist:
            return math.inf, [], 0.0, math.inf
        # rebuild path
        path = [dst]
        bottleneck = math.inf
        cost_sum = 0.0
        cur = dst
        while cur != src:
            pu, e = prev[cur]
            path.append(pu)
            bottleneck = min(bottleneck, e.capacity_gbps)
            cost_sum += e.cost_per_GB
            cur = pu
        path.reverse()
        return dist[dst] / 1000.0, path, (0.0 if bottleneck is math.inf else bottleneck), cost_sum
