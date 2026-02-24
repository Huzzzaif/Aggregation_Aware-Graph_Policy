# Aggregation-Aware Graph Policy Federated Learning Simulation

A SimPy-based simulation of a **privacy-preserving Federated Learning (FL)** pipeline over a realistic network topology. Plaintext model weights are trained on-device, encrypted at the edge using **CKKS homomorphic encryption** (TenSEAL), routed through an **In-Network Computing (INC)** overlay for homomorphic aggregation, and consumed by an **encrypted server**  no raw weights ever leave the device.

---

## Architecture Overview

```
┌──────────────┐        WEIGHTS        ┌──────────────┐      HE_WEIGHTS      ┌─────────────────┐
│  SmartWatch  │ ───────────────────►  │  Edge Router │ ───────────────────► │   INC Overlay   │
│  (Device)    │   plaintext (local    │  (Encrypts   │    CKKS ciphertext   │  (Homomorphic   │
│  trains on   │    model weights)     │  w/ CKKS)    │                      │   Aggregation)  │
│  CSV rows    │                       │              │                      │                 │
└──────────────┘                       └──────────────┘                      └────────┬────────┘
                                                                                       │ HE_GLOBAL_AGG
                                                                                       ▼
                                                                             ┌─────────────────┐
                                                                             │ Encrypted Server│
                                                                             │ (No secret key) │
                                                                             │ Homomorphic EMA │
                                                                             └─────────────────┘
```

**Key property:** The INC overlay and server never hold the secret key — they operate entirely on ciphertexts.

---

## File Structure

| File | Description |
|------|-------------|
| `smartWatch.py` | Simulates an Apple Watch Series 10–style IoT device. Trains a `LinearModel` via SGD on rows from a CSV, drains battery via screen/Wi-Fi power model, and periodically uploads plaintext weights. |
| `edgeDevice.py` | Edge router that receives plaintext weights, encrypts them using TenSEAL CKKS (owns the secret key), and forwards ciphertexts to the INC overlay. |
| `inc_network.py` | INC overlay nodes that forward or homomorphically aggregate CKKS ciphertexts without decrypting. Supports configurable routing policies and periodic flush to the server. |
| `server.py` | Encrypted FL server that updates a global model in-place using homomorphic EMA — no decryption, no secret key. |
| `FL_Loop.py` | End-to-end runner: wires all components together, runs the SimPy simulation, and prints energy summaries. |

> **Note:** This repo also expects `model.py` (LinearModel) and `net.py` (Packet, DummyNet, NetworkAPI) which are not listed above — ensure they are present before running.

---

## Packet Flow

| Step | Packet Kind | From → To | Payload |
|------|------------|-----------|---------|
| 1 | `WEIGHTS` | Device → Router | Plaintext model bytes |
| 2 | `HE_WEIGHTS` | Router → INC node | CKKS ciphertext bytes |
| 3 | `HE_WEIGHTS_FWD` | INC → INC (multi-hop) | CKKS ciphertext bytes |
| 4 | `HE_GLOBAL_AGG` | INC aggregator → Server | Aggregated CKKS ciphertext bytes |

---

## Requirements

```bash
pip install simpy tenseal
```

Python 3.9+ recommended.

---

## Quickstart

1. **Prepare your CSV** (must have a header row). The default config expects:
   ```
   participant_id, date, age, gender, ..., avg_heart_rate, bmi, hydration_level, ...
   ```

2. **Set your CSV path and columns** in `FL_Loop.py`:
   ```python
   csv_path = "smartwatch_data.csv"
   features = ["duration_minutes", "bmi", "hydration_level"]
   target   = "avg_heart_rate"
   ```

3. **Run the simulation:**
   ```bash
   python FL_Loop.py
   ```

4. **Output** — energy summaries per component and encrypted global model size:
   ```
   Router energy: {'wifi_j': ..., 'cpu_j': ..., 'total_Wh': ...}
   INC_AGG  energy: {...}
   Server encrypted global version: 18
   Server encrypted global size: 131072 bytes
   ```

---

## Configuration

### Device (`smartWatch.py` / `FL_Loop.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `battery_Wh` | 1.266 | Apple Watch Series 10 ~46mm battery |
| `train_interval_s` | 2.0 | SGD step every N seconds |
| `wifi_sync_period_s` | 600.0 | Upload weights every 10 min |
| `learning_rate` | 0.05 | SGD learning rate |
| `encoders` | `{}` | Label encoders for categorical columns |

### Router (`edgeDevice.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poly_mod_degree` | 8192 | CKKS polynomial modulus degree |
| `coeff_mod_bit_sizes` | `[60,40,40,60]` | CKKS coefficient modulus |
| `global_scale` | `2**40` | CKKS scale |

### INC (`inc_network.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `is_aggregator` | `False` | Designate this node as the aggregator |
| `agg_period_s` | 300.0 | Flush aggregate to server every 5 min |
| `agg_average` | `True` | Average the batch (vs. raw sum) |
| `routing_policy` | `shortest_queue_policy` | Pluggable routing function |

### Server (`server.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eta` | 1.0 | EMA step size: `G ← (1-η)G + η·U` |

---

## Adding More Devices

Instantiate additional `Device` objects with unique IDs pointing to the same router:

```python
dev2 = Device(env, net, device_id="watch02", router_id="routerA", ...)
dev3 = Device(env, net, device_id="watch03", router_id="routerA", ...)
```

---

## Multi-Hop INC Topology

Change `inc_id` in `Router` to route through a forwarding node first:

```python
router = Router(..., inc_id="INC_FWD1", ...)
# INC_FWD1 -> INC_AGG -> SERVER
```

---

## Privacy Guarantee

- The **secret key never leaves the router**.
- INC nodes and the server operate on **ciphertexts only** using TenSEAL's eval-only context.
- The global model stored on the server is **never decrypted** in this simulation.
- To decrypt for evaluation, only the router (or a key holder) can call `ct.decrypt()`.

---

## Energy Accounting

Each component tracks coarse-grained energy consumption (Wi-Fi + CPU in Joules). These are order-of-magnitude estimates for research/paper use — not hardware-calibrated measurements.

---

## License

MIT
