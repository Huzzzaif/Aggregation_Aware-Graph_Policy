# run.py
#
# End-to-end FL loop (encrypted, no decryption on the path):
# Device (plaintext weights) -> Router (CKKS encrypt) -> INC overlay (route+aggregate) -> Server (updates encrypted global)
#
# Requirements:
#   pip install simpy tenseal
#
# Project structure this expects:
#   model.py          (LinearModel)
#   net.py            (Packet, DummyNet)
#   device.py         (Device, Series10Profile)
#   router.py         (Router, RouterProfile)            # router owns CKKS secret
#   inc.py            (INCNode, INCProfile)              # eval-only context (no secret)
#   server.py         (EncryptedServer, ServerProfile)   # eval-only context (no secret)
#
# Notes:
# - Router generates a CKKS context with a secret key.
# - We create a **public/eval** clone of that context (no secret) for INC + Server
#   via serialize(save_secret_key=False) -> load.
# - INC aggregator homomorphically averages ciphertexts and forwards to Server.
# - Server updates the encrypted global model (homomorphic EMA), STILL encrypted.

from __future__ import annotations
import simpy
import tenseal as ts

from net import DummyNet
from device import Device, Series10Profile
from router import Router, RouterProfile
from inc import INCNode, INCProfile
from server import EncryptedServer, ServerProfile


def clone_public_context(ctx_with_secret: ts.Context) -> ts.Context:
    """Create a TenSEAL context WITHOUT secret key from an existing one."""
    blob = ctx_with_secret.serialize(save_secret_key=False)
    return ts.context_from(blob)


def main():
    # ---------- Sim & network ----------
    env = simpy.Environment()
    net = DummyNet(env, link_mbps=18.0, base_latency_ms=8.0)

    # ---------- Router (edge device) with secret CKKS context ----------
    def rlog(msg: str): print(msg)
    router = Router(
        env, net,
        router_id="routerA", inc_id="INC_AGG",
        profile=RouterProfile(),
        log=rlog,
        # (Router will create its own CKKS context if not provided)
        poly_mod_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        global_scale=2**40,
    )

    # Produce a public/eval-only context for INC & Server
    public_ctx = clone_public_context(router.ctx)

    # ---------- INC overlay ----------
    # Topology (example):
    #   routerA -> INC_FWD1 -> INC_AGG (aggregator) -> SERVER
    #
    # You can add more FWD nodes or change neighbors to test different overlays.
    def ilog(msg: str): print(msg)
    fwd = INCNode(
        env, net,
        node_id="INC_FWD1",
        neighbors=["INC_AGG"],           # forward toward aggregator
        server_id="SERVER",              # server sink
        ts_ctx_public=public_ctx,        # no secret key
        profile=INCProfile(),
        is_aggregator=False,
        routing_policy=None,             # default: shortest_queue_policy
        log=ilog,
    )

    agg = INCNode(
        env, net,
        node_id="INC_AGG",
        neighbors=["SERVER"],            # for non-agg traffic (unused here)
        server_id="SERVER",
        ts_ctx_public=public_ctx,        # no secret key
        profile=INCProfile(),
        is_aggregator=True,              # <- designated aggregator
        agg_period_s=300.0,              # flush every 5 min (tune as you like)
        agg_average=True,                # average the batch
        log=ilog,
    )

    # ---------- Encrypted Server (NO secret key; homomorphic update only) ----------
    def slog(msg: str): print(msg)
    server = EncryptedServer(
        env, net,
        server_id="SERVER",
        ts_public_ctx=public_ctx,        # eval-only context
        eta=1.0,                         # 1.0 = replace with incoming aggregate
        profile=ServerProfile(),
        log=slog,
    )

    # ---------- Device (watch) ----------
    # CSV: choose features/target available in your data.
    # From your header example:
    #   participant_id,date,age,gender,height_cm,weight_kg,activity_type,duration_minutes,intensity,calories_burned,
    #   avg_heart_rate,hours_sleep,stress_level,daily_steps,hydration_level,bmi,resting_heart_rate,
    #   blood_pressure_systolic,blood_pressure_diastolic,health_condition,smoking_status,fitness_level
    #
    # Example: predict avg_heart_rate from duration_minutes, bmi, hydration_level.
    csv_path = "smartwatch_data.csv"  # <-- set to your CSV path
    features = ["duration_minutes", "bmi", "hydration_level"]
    target = "avg_heart_rate"

    # Optional simple encoders if you include categoricals (here we don't use them)
    encoders = {
        # "intensity": {"Low": 0.0, "Moderate": 1.0, "High": 2.0},
        # "gender": {"M": 0.0, "F": 1.0},
    }

    dev = Device(
        env, net,
        device_id="watch01", router_id="routerA",
        profile=Series10Profile(
            battery_Wh=1.266,
            display_on_W=0.6,
            display_off_W=0.02,
            display_on_min_per_hour=8.0,
            wifi_idle_W=0.05,
            wifi_tx_W=0.6,
            wifi_tx_burst_s=2.0,
            wifi_sync_period_s=600.0,     # upload local weights every 10 minutes
            wifi_pull_period_s=600.0,     # (unused in this loop; pull is for global->device)
        ),
        csv_path=csv_path,
        feature_columns=features,
        target_column=target,
        learning_rate=0.05,
        train_interval_s=2.0,            # train on one row every 2 seconds
        sync_period_s=600.0,             # upload to router every 10 minutes
        pull_period_s=None,              # we’re not pulling global to device in this run
        encoders=encoders,
        preprocess=None,
        decrypt_fn=None,                 # not needed (device only uploads plaintext)
    )

    # ---------- Wire: router -> INC_FWD1 ----------
    # By default, Router forwards to inc_id="INC_AGG". If you want an extra hop via INC_FWD1:
    # Make the router forward to INC_FWD1 instead of INC_AGG by changing inc_id above.
    # We'll keep Router -> INC_AGG for single-hop into INC. To test multi-hop, set:
    #   router = Router(..., inc_id="INC_FWD1", ...)
    #
    # If you want multi-device, instantiate more Device(...) with different IDs.

    # ---------- Run the sim ----------
    sim_hours = 3.0
    env.run(until=sim_hours * 3600.0)

    # ---------- Summaries ----------
    print("\n=== DONE ===")
    # Router energy
    try:
        print("Router energy:", router.energy_summary())
    except Exception:
        pass
    # INC nodes energy
    try:
        print("INC_FWD1 energy:", fwd.energy_summary())
        print("INC_AGG  energy:", agg.energy_summary())
    except Exception:
        pass
    # Server encrypted version & energy
    try:
        print(f"Server encrypted global version: {server.version}")
        enc_blob = server.encrypted_global_bytes()
        print(f"Server encrypted global size: {len(enc_blob) if enc_blob else 0} bytes")
        print("Server energy:", server.energy_summary())
    except Exception:
        pass


if __name__ == "__main__":
    main()