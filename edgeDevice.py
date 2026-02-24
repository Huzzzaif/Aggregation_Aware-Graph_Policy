"""
router.py

Edge Router (Wi-Fi only) that:
  1) receives plaintext model weights from devices (kind="WEIGHTS"),
  2) encrypts them with **TenSEAL CKKS** using a router-owned key/context,
  3) forwards ciphertext to the cloud/INC (kind="HE_WEIGHTS"),
  4) tracks simple Wi-Fi + CPU energy usage.

Dependencies:
  pip install tenseal simpy

Assumes you already have:
  - model.py  (LinearModel with to_bytes/from_bytes helpers)
  - net.py    (Packet + DummyNet / NetworkAPI)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import simpy
import tenseal as ts

from net import Packet, NetworkAPI
from model import LinearModel


# -----------------------------------------------------------------------------
# Router energy profile (very rough defaults; tune to your setup)
# -----------------------------------------------------------------------------
@dataclass
class RouterProfile:
    # Power model (Watts)
    wifi_idle_W: float = 1.5     # idle baseline for Wi-Fi
    wifi_rx_W: float = 2.0       # active receive window
    wifi_tx_W: float = 3.0       # active transmit window
    cpu_active_W: float = 5.0    # CPU while encrypting

    # Accounting window around packets (seconds)
    rx_active_floor_s: float = 0.010
    tx_active_floor_s: float = 0.010


# -----------------------------------------------------------------------------
# Router with TenSEAL CKKS encryption
# -----------------------------------------------------------------------------
class Router:
    """
    Minimal edge router that *owns* the TenSEAL context/keys and performs CKKS
    encryption on incoming model weights before forwarding to INC/cloud.

    Protocol
    --------
    device -> router: Packet(kind="WEIGHTS", payload=plaintext bytes)
    router -> INC   : Packet(kind="HE_WEIGHTS", payload=ciphertext bytes)

    TenSEAL context
    ---------------
    - We generate a CKKS context with a standard parameter set.
    - Encryption uses `ts.ckks_vector(context, floats).serialize()`.
    - Only the router holds the secret key; the INC receives ciphertext bytes.
      (If you later need the INC to *operate* on ciphertext, send it a
       copy of the context *without* the secret key plus relin/galois keys.)
    """

    def __init__(
        self,
        env: simpy.Environment,
        net: NetworkAPI,
        *,
        router_id: str = "routerA",
        inc_id: str = "INC",
        profile: Optional[RouterProfile] = None,
        log: Optional[Callable[[str], None]] = None,
        # If you already created a TenSEAL context elsewhere, pass it in
        ts_context: Optional[ts.Context] = None,
        # CKKS params (used only if ts_context is None)
        poly_mod_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[list[int]] = None,  # e.g., [60, 40, 40, 60]
        global_scale: float = 2**40,
    ) -> None:
        self.env = env
        self.net = net
        self.id = router_id
        self.inc_id = inc_id
        self.inbox: simpy.Store = net.register(self.id)
        self.inc_inbox: simpy.Store = net.register(self.inc_id)

        self.p = profile or RouterProfile()
        self.log = log or (lambda s: None)

        # Energy counters (J)
        self.wifi_j = 0.0
        self.cpu_j = 0.0

        # --- TenSEAL setup (router holds the key) ---
        if ts_context is None:
            # Reasonable CKKS defaults for small vectors of weights
            if coeff_mod_bit_sizes is None:
                coeff_mod_bit_sizes = [60, 40, 40, 60]
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_mod_degree=poly_mod_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            ctx.global_scale = global_scale
            ctx.generate_galois_keys()
            ctx.generate_relin_keys()
            # Keep the secret key on the router; do NOT disable it here.
            self.ctx = ctx
        else:
            self.ctx = ts_context

        # Background processes
        env.process(self._wifi_idle_loop())
        env.process(self._inbox_loop())

    # -------------------- energy helpers --------------------
    def _spend_wifi(self, seconds: float, watts: float) -> None:
        if seconds <= 0 or watts <= 0:
            return
        self.wifi_j += watts * seconds

    def _spend_cpu(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.cpu_j += self.p.cpu_active_W * seconds

    # -------------------- processes ------------------------
    def _wifi_idle_loop(self):
        """Account idle Wi-Fi drain per second."""
        while True:
            self._spend_wifi(1.0, self.p.wifi_idle_W)
            yield self.env.timeout(1.0)

    def _inbox_loop(self):
        """Encrypt incoming weights with CKKS and forward to INC."""
        while True:
            pkt: Packet = yield self.inbox.get()
            if pkt.kind != "WEIGHTS":
                continue

            # RX active window (coarse)
            self._spend_wifi(self.p.rx_active_floor_s, self.p.wifi_rx_W)

            # Parse plaintext weights -> float list
            try:
                model = LinearModel.from_bytes(pkt.payload)
                vec = model.w + [model.b]
            except Exception as e:
                self.log(f"[{self.env.now:7.3f}s] Router: failed to parse weights: {e}")
                continue

            # Encrypt using TenSEAL CKKS
            # We simulate CPU energy spent during encryption by measuring elapsed sim time.
            t0 = self.env.now
            ct_bytes = yield self._encrypt_async(vec)
            t1 = self.env.now
            self._spend_cpu(t1 - t0)

            # Build outbound packet to INC/cloud
            out = Packet(
                src=self.id,
                dst=self.inc_id,
                kind="HE_WEIGHTS",
                payload=ct_bytes,
                birth_time_s=self.env.now,
            )

            # TX active window (coarse)
            self._spend_wifi(self.p.tx_active_floor_s, self.p.wifi_tx_W)
            yield self.net.send(out)

            self.log(
                f"[{self.env.now:7.3f}s] Router: encrypted weights (len={len(vec)}) "
                f"-> ciphertext {len(ct_bytes)/1024:.2f} KiB; "
                f"CPU_j={self.cpu_j:.3f} WiFi_j={self.wifi_j:.3f}"
            )

    # -------------------- TenSEAL async wrapper --------------
    def _encrypt_async(self, floats: list[float]) -> simpy.events.Event:
        """
        Wrap TenSEAL encryption so it plays nicely with SimPy's event loop.
        We yield once to simulate minimal scheduling delay; the crypto call is
        synchronous but fast enough for small vectors.
        """
        def _job():
            # Give control back to the scheduler once (optional)
            yield self.env.timeout(0)
            # CKKS encrypt -> serialized bytes
            ctxt = ts.ckks_vector(self.ctx, floats)
            blob = ctxt.serialize()
            # Yield again to finish
            yield self.env.timeout(0)
            return blob

        return self.env.process(_job())

    # -------------------- summary ----------------------------
    def energy_summary(self) -> dict:
        total = self.wifi_j + self.cpu_j
        return {
            "wifi_j": round(self.wifi_j, 6),
            "cpu_j": round(self.cpu_j, 6),
            "total_j": round(total, 6),
            "total_Wh": round(total / 3600.0, 6),
        }