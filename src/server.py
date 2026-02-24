"""
server.py

Encrypted FL Server (no decryption)
===================================

Receives *aggregated CKKS-encrypted* model updates from INC (kind="HE_GLOBAL_AGG")
and updates an **encrypted** global model in-place using homomorphic ops only.

Key points
----------
- No secret key here. The server holds a **public/eval TenSEAL context**
  (no secret key) so it can add / multiply by plaintext scalars, but cannot decrypt.
- Global model is stored as a CKKS ciphertext vector (weights + bias).
- Update rule (homomorphic EMA by default):
      G <- (1 - eta) * G  +  eta * U
  where:
      G : current encrypted global model (CKKSVector)
      U : incoming aggregated encrypted weights (CKKSVector)
      eta : server-side step size in (0, 1]
- If G is not initialized, it will be set to the first incoming aggregate.

Dependencies
------------
pip install simpy tenseal
Local modules:
  - net.Packet, net.NetworkAPI
Packet kind consumed:
  - "HE_GLOBAL_AGG" : payload = serialized CKKS ciphertext (weights+bias)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import simpy
import tenseal as ts

from net import Packet, NetworkAPI


# ---------------------------------------------------------------------------
# Optional: coarse energy model (can be removed if not needed)
# ---------------------------------------------------------------------------
@dataclass
class ServerProfile:
    net_rx_W: float = 8.0         # active receive window (Watts)
    cpu_active_W: float = 15.0    # homomorphic ops (Watts)
    rx_active_floor_s: float = 0.010


# ---------------------------------------------------------------------------
# Encrypted FL Server (no decryption)
# ---------------------------------------------------------------------------
class EncryptedServer:
    """
    Holds an **encrypted** global model and updates it homomorphically without decrypting.

    Parameters
    ----------
    env : simpy.Environment
    net : NetworkAPI
    server_id : str
    ts_public_ctx : ts.Context
        TenSEAL CKKS context with *no secret key* (public/eval only). Create from
        a secret-bearing context via: ctx.make_context_public()
    eta : float
        Step size for the global update: G <- (1 - eta) * G + eta * U
    profile : ServerProfile (optional)
    log : Callable[[str], None] (optional)
    """

    def __init__(
        self,
        env: simpy.Environment,
        net: NetworkAPI,
        *,
        server_id: str = "SERVER",
        ts_public_ctx: ts.Context,
        eta: float = 1.0,                         # 1.0 = replace with incoming aggregate
        profile: Optional[ServerProfile] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.env = env
        self.net = net
        self.id = server_id
        self.inbox: simpy.Store = net.register(self.id)

        self.ctx = ts_public_ctx                 # no secret key!
        self.eta = float(eta)
        assert 0.0 < self.eta <= 1.0, "eta must be in (0, 1]"

        self.profile = profile or ServerProfile()
        self.log = log or (lambda s: None)

        # Encrypted global model state (CKKSVector) + version counter
        self._global_ct: Optional[ts.CKKSVector] = None
        self.version: int = 0

        # Energy accounting (optional)
        self.net_j = 0.0
        self.cpu_j = 0.0

        # Start processing loop
        env.process(self._inbox_loop())

    # ------------------------ energy helpers (optional) ------------------------
    def _spend_rx(self) -> None:
        self.net_j += self.profile.net_rx_W * self.profile.rx_active_floor_s

    def _spend_cpu(self, seconds: float) -> None:
        if seconds > 0:
            self.cpu_j += self.profile.cpu_active_W * seconds

    # ----------------------------- main loop ----------------------------------
    def _inbox_loop(self):
        while True:
            pkt: Packet = yield self.inbox.get()
            if pkt.kind != "HE_GLOBAL_AGG":
                continue

            # Account a small RX window
            self._spend_rx()

            # Deserialize incoming aggregated ciphertext
            try:
                t0 = self.env.now
                incoming_ct = ts.ckks_vector_from(self.ctx, pkt.payload)
            except Exception as e:
                self.log(f"[{self.env.now:7.3f}s] Server: failed to deserialize agg ciphertext: {e}")
                continue

            # If first update, adopt it as the global model; else homomorphic EMA
            try:
                if self._global_ct is None:
                    # G <- U
                    self._global_ct = incoming_ct
                else:
                    # G <- (1 - eta) * G + eta * U
                    # multiply by plaintext scalars (supported without secret key)
                    one_minus_eta = 1.0 - self.eta
                    # Important: keep operations minimal to avoid scale exhaustion for long runs
                    self._global_ct = (self._global_ct * one_minus_eta) + (incoming_ct * self.eta)
                t1 = self.env.now
                self._spend_cpu(t1 - t0)
            except Exception as e:
                self.log(f"[{self.env.now:7.3f}s] Server: homomorphic update failed: {e}")
                continue

            self.version += 1
            self.log(f"[{self.env.now:7.3f}s] Server: updated encrypted global v{self.version}")

    # ----------------------------- accessors ----------------------------------
    def encrypted_global_bytes(self) -> Optional[bytes]:
        """
        Returns the serialized **encrypted** global model (or None if uninitialized).
        You can forward this to routers/devices/INC; they still can't decrypt it
        unless they hold the secret key.
        """
        if self._global_ct is None:
            return None
        return self._global_ct.serialize()

    def energy_summary(self) -> dict:
        total = self.net_j + self.cpu_j
        return {
            "net_j": round(self.net_j, 6),
            "cpu_j": round(self.cpu_j, 6),
            "total_j": round(total, 6),
            "total_Wh": round(total / 3600.0, 6),
        }