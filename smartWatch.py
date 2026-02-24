#This class will modela device from which the heathcare data will be injected into the network.
#It will be modelled behind the apple watch series 10, which is currently the most used smartwatch in the USA Sept 2025
"""
device.py

Minimal Apple Watch Series 10–style Device (only what we agreed):
- Battery (Wh)
- Screen duty-cycle (on/off)
- Wi-Fi comms (idle draw + periodic sync sending current model weights)
- Local SGD on rows from an external CSV every T seconds

Everything else is intentionally omitted.

This module depends on:
  - model.LinearModel (tiny y = w·x + b with SGD + (de)serialization)
  - net.Packet, net.NetworkAPI (mailboxes + send)

You control which CSV columns become features and which single column is the target.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Iterator, Iterable, List, Tuple, Optional, Dict, Callable

import simpy


# ---------------------------------------------------------------------
# Series 10 profile: just the knobs we need for this minimal simulator
# ---------------------------------------------------------------------
@dataclass
class Series10Profile:
    battery_Wh: float = 1.266               # approx 46 mm class
    display_on_W: float = 0.6               # active screen (average)
    display_off_W: float = 0.02             # AOD / dimmed
    display_on_min_per_hour: float = 8.0    # ~8 minutes per hour visible
    wifi_idle_W: float = 0.05               # background Wi-Fi draw
    wifi_tx_W: float = 0.6                  # TX burst power during sync
    wifi_sync_period_s: float = 600.0       # sync to router every 10 minutes
    wifi_tx_burst_s: float = 2.0            # assume 2s active TX per sync


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
class Device:
    """
    A tiny SimPy actor that:
      (1) drains battery based on screen duty and Wi-Fi,
      (2) trains a LinearModel from an external CSV every `train_interval_s`,
      (3) periodically sends current weights to the router via Wi-Fi.

    Notes:
    - CSV must have a header row. You choose feature/target columns by name.
    - Basic categorical mapping helpers are provided for common fields
      (like "intensity", "gender"). Extend `encoders` as needed.

    Example (runner):
        dev = Device(
            env, net, device_id="watch01", router_id="routerA",
            profile=Series10Profile(),
            csv_path="smartwatch_data.csv",
            feature_columns=["duration_minutes","bmi","hydration_level"],  # X
            target_column="avg_heart_rate",                                # y
            learning_rate=0.05,
            train_interval_s=2.0,
            encoders={"intensity": {"Low": 0, "Moderate": 1, "High": 2},
                      "gender": {"M": 0, "F": 1}},
        )
    """

    def __init__(
        self,
        env: simpy.Environment,
        device_id: str,
        *,
        router_id: str,
        profile: Series10Profile,
        csv_path: str,
        feature_columns: List[str],          # names of X columns in CSV header
        target_column: str,                  # name of y column in CSV header
        learning_rate: float = 0.05,
        train_interval_s: float = 2.0,
        sync_period_s: Optional[float] = None,
        encoders: Optional[Dict[str, Dict[str, float]]] = None,  # simple label encoders
        preprocess: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
    ) -> None:

        # --- power model / battery ---
        self.p = profile
        self.soc_Wh = float(self.p.battery_Wh)

        # CSV row generator (infinite, cycles over file)
        self._row_iter = self._csv_rows(csv_path)

        # --- background processes ---
        env.process(self._screen_loop())
        env.process(self._wifi_idle_loop())
        env.process(self._train_loop())
        env.process(self._sync_loop())

    # ---------------------- battery helper ----------------------
    def _draw(self, watts: float, seconds: float) -> None:
        """Battery drain = power * time (capped at 0)."""
        if watts <= 0 or seconds <= 0 or self.soc_Wh <= 0:
            return
        self.soc_Wh -= watts * seconds / 3600.0
        if self.soc_Wh < 0:
            self.soc_Wh = 0.0

    # ---------------------- processes ---------------------------
    def _screen_loop(self):
        """Each simulated hour: ON for N minutes, otherwise OFF (AOD)."""
        on_s = max(0.0, self.p.display_on_min_per_hour * 60.0)
        off_s = max(0.0, 3600.0 - on_s)
        while self.soc_Wh > 0:
            # active period
            self._draw(self.p.display_on_W, on_s)
            yield self.env.timeout(on_s)
            # AOD / dim period
            self._draw(self.p.display_off_W, off_s)
            yield self.env.timeout(off_s)

    def _wifi_idle_loop(self):
        """Account Wi-Fi idle draw continuously (simple 1-second quanta)."""
        while self.soc_Wh > 0:
            self._draw(self.p.wifi_idle_W, 1.0)
            yield self.env.timeout(1.0)

    def _train_loop(self):
        """Every train_interval_s: one SGD step from the next CSV row."""
        while self.soc_Wh > 0:
            row = next(self._row_iter)  # raw dict of string values (header->value)
            # optional user-supplied preprocessor (e.g., scaling/cleaning)
            if self.preprocess:
                row = self.preprocess(row)

            # build numeric feature vector x and target y
            x = [self._to_float(col, row.get(col, "")) for col in self.feature_columns]
            y = self._to_float(self.target_column, row.get(self.target_column, ""))

            # one SGD step on y = w·x + b with squared error
            self.model.sgd_step(x, y, self.lr)

            # wait until next tick
            yield self.env.timeout(self.train_interval_s)

    def _sync_loop(self):
        """Every sync_period_s: send current weights to router via Wi-Fi."""
        while self.soc_Wh > 0:
            yield self.env.timeout(self.sync_period_s)

            # account a TX burst (simple constant-time model)
            self._draw(self.p.wifi_tx_W, self.p.wifi_tx_burst_s)

            pkt = Packet(
                src=self.id,
                dst=self.router_id,
                kind="WEIGHTS",
                payload=self.model.to_bytes(),
                birth_time_s=self.env.now,
            )
            # fire-and-forget; the router will consume from its mailbox
            yield self.net.send(pkt)

    # ---------------------- CSV handling ------------------------
    def _csv_rows(self, path: str) -> Iterator[Dict[str, str]]:
        """
        Infinite generator of dict rows (header -> value).
        - Requires header.
        - If file ends, restart from the beginning.
        - Skips empty lines.
        """
        cached: List[Dict[str, str]] = []

        def load_once() -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            with open(path, "r", newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    raise ValueError("CSV needs a header row.")
                for row in r:
                    if not row:
                        continue
                    out.append(row)
            return out

        while True:
            if not cached:
                cached = load_once()
                if not cached:
                    # if file empty, produce dummy zeros so the sim keeps moving
                    yield {c: "0" for c in (self.feature_columns + [self.target_column])}
                    continue
            for row in cached:
                yield row

    # ---------------------- value parsing -----------------------
    def _to_float(self, col: str, raw: str) -> float:
        """
        Convert a CSV cell to float with a few conveniences:
          - If column has an encoder (categorical -> numeric), use it.
          - Strip units like 'cm', 'kg' if present (best-effort).
          - Coerce blank/invalid to 0.0.
        """
        # categorical mapping (e.g., intensity: Low/Moderate/High)
        if col in self.encoders:
            mapping = self.encoders[col]
            if raw in mapping:
                return float(mapping[raw])
            # unseen label -> 0.0 (or extend: mapping.get(raw, default))
            return 0.0

        # best-effort unit stripping for common fields
        s = (raw or "").strip()
        for suffix in ("cm", "kg"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        try:
            return float(s)
        except Exception:
            return 0.0