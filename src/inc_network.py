"""
inc.py

INC (in-network computing) overlay that:
  • receives CKKS-encrypted weight packets from the router (kind="HE_WEIGHTS"),
  • forwards them hop-by-hop across an INC graph using a pluggable routing policy,
  • (optionally) homomorphically aggregates at a designated aggregator node, and
  • forwards the aggregated ciphertext to a server sink.

Assumptions
-----------
• Transport is your existing mailbox-style NetworkAPI (e.g., net.DummyNet).
• Ciphertexts are TenSEAL CKKS-serialized bytes created by the Router.
• INC nodes hold a TenSEAL context with **no secret key** (public/eval keys only)
  so they can do homomorphic ops (add, multiply_plain) but cannot decrypt.
• Only the final **Server** (not modeled here) or some trusted endpoint holds
  the secret key to decrypt the aggregated ciphertext.

Packet kinds
------------
FROM router  -> INC node(s):  kind="HE_WEIGHTS"        payload = CKKS ciphertext bytes
INC forward  -> INC node(s):  kind="HE_WEIGHTS_FWD"    payload = CKKS ciphertext bytes
Aggregator   -> Server sink :  kind="HE_GLOBAL_AGG"     payload = CKKS aggregated ciphertext bytes

You can add simple metadata in the `meta` header (first line), format:
  "round=<int>;client=<id>"

Minimal energy model
--------------------
Each INC node tracks coarse Wi-Fi energy for idle/RX/TX and CPU energy while
doing homomorphic ops. These are order-of-magnitude placeholders for papers.

Dependencies
------------
pip install simpy tenseal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Tuple
import simpy
import tenseal as ts

from net import Packet, NetworkAPI


# ---------------------------------------------------------------------------
# Energy profile for INC nodes
# ---------------------------------------------------------------------------
@dataclass
class INCProfile:
    # Wi-Fi-like power (Watts)
    wifi_idle_W: float = 1.2
    wifi_rx_W: float = 1.8
    wifi_tx_W: float = 2.5
    # CPU while evaluating homomorphic ops
    cpu_active_W: float = 4.0
    # Accounting windows around packets (seconds)
    rx_active_floor_s: float = 0.008
    tx_active_floor_s: float = 0.008


# ---------------------------------------------------------------------------
# Routing policy signature
# ---------------------------------------------------------------------------
RoutingPolicy = Callable[
    [str, List[str], Packet, "INCNode"],  # (this_node_id, neighbor_ids, pkt, node)
    str                                   # returns next_hop_id
]


def shortest_queue_policy(this_id: str, neighbors: List[str], pkt: Packet, node: "INCNode") -> str:
    """
    Pick the neighbor with the smallest mailbox length (very simple congestion-aware).
    Falls back to the first neighbor if equal.
    """
    if not neighbors:
        return this_id  # nowhere to go (will get dropped by transport if dst unknown)
    best = neighbors[0]
    best_len = node.net_len(best)
    for n in neighbors[1:]:
        ln = node.net_len(n)
        if ln < best_len:
            best, best_len = n, ln
    return best


# ---------------------------------------------------------------------------
# INC Node
# ---------------------------------------------------------------------------
class INCNode:
    """
    A single INC node in the overlay.
    • If is_aggregator=True, it accumulates ciphertexts and periodically forwards the
      homomorphic sum (or average) to the server sink.
    • Otherwise, it forwards packets according to a routing policy.

    ts_ctx_public: a TenSEAL CKKS context **without a secret key** (public/eval only).
                   You can obtain it on the Router via: ctx.make_context_public()
    """

    def __init__(
        self,
        env: simpy.Environment,
        net: NetworkAPI,
        *,
        node_id: str,
        neighbors: List[str],                 # static neighbor list for this node
        server_id: str,                       # where the final aggregate goes
        ts_ctx_public: ts.Context,            # no secret key; eval ops only
        profile: Optional[INCProfile] = None,
        is_aggregator: bool = False,
        agg_period_s: float = 300.0,          # flush every 5 minutes by default
        agg_average: bool = True,             # average (sum * 1/N) vs raw sum
        routing_policy: RoutingPolicy = shortest_queue_policy,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.env = env
        self.net = net
        self.id = node_id
        self.server_id = server_id
        self.inbox: simpy.Store = net.register(self.id)
        # NOTE: We assume net exposes a mailbox for neighbors already (via register elsewhere)
        self.neighbors = list(neighbors)
        self.p = profile or INCProfile()
        self.is_aggregator = bool(is_aggregator)
        self.agg_period_s = float(agg_period_s)
        self.agg_average = bool(agg_average)
        self.route = routing_policy
        self.log = log or (lambda s: None)

        # TenSEAL public/eval context (no secret key here)
        self.ctx = ts_ctx_public

        # Energy accounting (J)
        self.wifi_j = 0.0
        self.cpu_j = 0.0

        # Aggregation state (only used on aggregator)
        self._sum_ct: Optional[ts.CKKSVector] = None
        self._count: int = 0

        # Background processes
        env.process(self._wifi_idle_loop())
        env.process(self._inbox_loop())
        if self.is_aggregator:
            env.process(self._flush_loop())

    # -------------- helper: neighbor mailbox length (for policy) --------------
    def net_len(self, node_id: str) -> int:
        """Best-effort: try to peek neighbor mailbox length if transport exposes it."""
        mbx = getattr(self.net, "mailboxes", None)
        if isinstance(mbx, dict) and node_id in mbx:
            q = mbx[node_id]
            try:
                return len(q.items)  # simpy.Store has .items list
            except Exception:
                return 0
        return 0

    # ------------------------ energy helpers ------------------------
    def _spend_wifi(self, seconds: float, watts: float) -> None:
        if seconds > 0 and watts > 0:
            self.wifi_j += watts * seconds

    def _spend_cpu(self, seconds: float) -> None:
        if seconds > 0:
            self.cpu_j += self.p.cpu_active_W * seconds

    # ------------------------ processes -----------------------------
    def _wifi_idle_loop(self):
        while True:
            self._spend_wifi(1.0, self.p.wifi_idle_W)
            yield self.env.timeout(1.0)

    def _inbox_loop(self):
        while True:
            pkt: Packet = yield self.inbox.get()

            if pkt.kind not in ("HE_WEIGHTS", "HE_WEIGHTS_FWD"):
                # Unknown traffic types ignored
                continue

            # RX accounting
            self._spend_wifi(self.p.rx_active_floor_s, self.p.wifi_rx_W)

            if self.is_aggregator:
                # Homomorphically add ciphertext to running sum
                try:
                    t0 = self.env.now
                    ct = ts.ckks_vector_from(self.ctx, pkt.payload)
                    # Charge CPU for homomorphic addition (very light)
                    if self._sum_ct is None:
                        self._sum_ct = ct
                    else:
                        self._sum_ct = self._sum_ct + ct
                    self._count += 1
                    t1 = self.env.now
                    self._spend_cpu(t1 - t0)
                    self.log(f"[{self.env.now:7.3f}s] INC[{self.id}] aggregated one (count={self._count})")
                except Exception as e:
                    self.log(f"[{self.env.now:7.3f}s] INC[{self.id}] agg error: {e}")
            else:
                # Forward to next hop toward the aggregator/server
                try:
                    nxt = self.route(self.id, self.neighbors, pkt, self)
                except Exception:
                    nxt = self.neighbors[0] if self.neighbors else self.server_id

                fwd = Packet(
                    src=self.id,
                    dst=nxt,
                    kind="HE_WEIGHTS_FWD",
                    payload=pkt.payload,          # still ciphertext bytes
                    birth_time_s=self.env.now,
                )
                # TX accounting
                self._spend_wifi(self.p.tx_active_floor_s, self.p.wifi_tx_W)
                yield self.net.send(fwd)

    def _flush_loop(self):
        """Aggregator: periodically forward the (sum or average) to the server."""
        while True:
            yield self.env.timeout(self.agg_period_s)
            if self._sum_ct is None or self._count == 0:
                continue

            try:
                t0 = self.env.now
                out_ct = self._sum_ct
                if self.agg_average and self._count > 0:
                    # Multiply by scalar (1/N) homomorphically
                    out_ct = out_ct * (1.0 / self._count)
                blob = out_ct.serialize()
                t1 = self.env.now
                self._spend_cpu(t1 - t0)

                pkt = Packet(
                    src=self.id,
                    dst=self.server_id,
                    kind="HE_GLOBAL_AGG",
                    payload=blob,
                    birth_time_s=self.env.now,
                )
                # TX accounting
                self._spend_wifi(self.p.tx_active_floor_s, self.p.wifi_tx_W)
                yield self.net.send(pkt)

                self.log(
                    f"[{self.env.now:7.3f}s] INC[{self.id}] flushed "
                    f"{'AVG' if self.agg_average else 'SUM'} of {self._count} -> server"
                )

                # Reset window
                self._sum_ct = None
                self._count = 0

            except Exception as e:
                self.log(f"[{self.env.now:7.3f}s] INC[{self.id}] flush error: {e}")

    # ------------------------ summary -------------------------------
    def energy_summary(self) -> dict:
        total = self.wifi_j + self.cpu_j
        return {
            "wifi_j": round(self.wifi_j, 6),
            "cpu_j": round(self.cpu_j, 6),
            "total_j": round(total, 6),
            "total_Wh": round(total / 3600.0, 6),
        }