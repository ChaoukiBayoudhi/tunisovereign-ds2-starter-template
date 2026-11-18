#!/usr/bin/env python3
"""
TuniChain-Sovereign - P3 synthetic dataset generator.

Generates, in the output directory (default: the folder of this script):
  1. tunichain_telemetry_dataset.csv  (main P3 dataset, 151,200 rows x 8 columns)
  2. tunichain_grpc_packets.csv       (long format, 453,600 rows x 6 columns)
  3. grpc_packet_example.json         (one TelemetryPacket example)

ALL DATA IS SYNTHETIC. It does not come from SONEDE, farms, farmers or any
official Tunisian institution.

Usage:
    python generate_dataset.py [--seed 42] [--out-dir DIR]

Dependencies: numpy, pandas (nothing else).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION (single block: every range / limit used by the generator)
# =============================================================================
SEED = 42
START_TS = "2026-11-10T00:00:00Z"
N_DAYS = 30
STEP_MINUTES = 10
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES          # 144
N_STEPS = N_DAYS * STEPS_PER_DAY                   # 4,320 per node
NODES_PER_GOVERNORATE = 5

# governorate -> (sensor_id prefix, mean November air temperature in C).
# Governorate spellings follow simulator/sensor_simulator.py ("Sidi Bouzid").
# The sensor_id prefixes for Kairouan/Tozeur/Sfax/Beja (KAI/TOZ/SFA/BEJ) are a
# P3 choice: the repo only fixes the pattern, not these prefixes.
GOVERNORATES = {
    "Kasserine":   ("KAS",  17.0),
    "Sidi Bouzid": ("SIDI", 18.0),
    "Gafsa":       ("GAF",  19.0),
    "Kairouan":    ("KAI",  18.5),
    "Tozeur":      ("TOZ",  20.0),
    "Sfax":        ("SFA",  20.0),
    "Beja":        ("BEJ",  15.5),
}
CROP = "OLIVE"   # all nodes are olive plots (ResourceNodePlot.crop_or_sector_type)

# --- Normal behaviour ---------------------------------------------------------
HUMIDITY_BASE = 58.0            # % (global baseline)
HUMIDITY_NODE_SD = 1.5          # per-node offset of the baseline
HUMIDITY_DIURNAL_AMP = 1.5      # % (small day/night swing)
HUMIDITY_DRIFT_SD = 1.5         # % (slow AR(1) drift, stationary std)
HUMIDITY_DRIFT_PHI = 0.998      # AR(1) coefficient per 10-minute step
HUMIDITY_NOISE_SD = 0.5         # %
HUMIDITY_IRRIGATION_GAIN = 2.0  # % added after pumping (exponentially smoothed)
HUMIDITY_IRRIGATION_TAU = 18    # steps (3 h) smoothing time constant

TEMP_DIURNAL_AMP = 6.0          # C, minimum ~03:00 UTC, maximum ~15:00 UTC
TEMP_WEATHER_SD = 1.5           # C, slow day-to-day weather AR(1) (stationary std)
TEMP_WEATHER_PHI = 0.999
TEMP_NOISE_SD = 0.4             # C

# Pump cycle (same windows as simulator/README_simulator.md: 06-10h and 18-20h).
PUMP_WINDOWS_HOURS = [(6, 10), (18, 20)]        # [start, end) in UTC hours
# water_flow unit: HYPOTHESIS H1 = L/min at sector level, to be confirmed by P2
# (the .proto carries no unit; TelemetryReading says "L/min or Units/h").
FLOW_ON_LEVEL = (140.0, 170.0)  # per-node ON flow, uniform range
FLOW_ON_NOISE_SD = 4.0
FLOW_OFF_LEVEL = 2.0            # residual flow when the pump is off
FLOW_OFF_NOISE_SD = 0.5

# --- Documented physical limits (hard clip + validation) --------------------
LIMITS = {
    "soil_humidity": (0.0, 100.0),   # %
    "temperature":   (-5.0, 55.0),   # C
    "water_flow":    (0.0, 400.0),   # L/min (H1)
}
NORMAL_HUMIDITY_EXPECTED = (45.0, 72.0)  # "generally" range for normal rows (checked, soft)

# --- Anomalies ----------------------------------------------------------------
ANOMALY_TYPES = [
    "low_soil_humidity",
    "high_flow_leak",
    "unauthorized_pumping",
    "sirocco_heat",
    "flow_drop",
]
ANOMALY_TARGET_RATE = 0.04      # ~4 % of rows (simulation choice, documented)
EPISODE_LEN = (3, 9)            # steps (30 to 90 minutes)
MIN_GAP_STEPS = 6               # minimum gap between two episodes of one node

# --- gRPC mapping (protos/tunisovereign.proto: TelemetryPacket) --------------
METRIC_ORDER = [("MOISTURE", "soil_humidity"), ("FLOW", "water_flow"), ("TEMP", "temperature")]
GRPC_COLUMNS = ["node_code", "governorate", "metric_type", "value",
                "iso_timestamp", "signature_hmac"]
MAIN_COLUMNS = ["sensor_id", "soil_humidity", "temperature", "water_flow",
                "timestamp", "governorate", "is_anomaly", "anomaly_type"]

MAIN_CSV = "tunichain_telemetry_dataset.csv"
GRPC_CSV = "tunichain_grpc_packets.csv"
JSON_EXAMPLE = "grpc_packet_example.json"


# =============================================================================
# 1. NODES AND TIME AXIS
# =============================================================================
def build_nodes():
    """Return a list of (sensor_id, governorate, mean_temp). Each node has exactly one governorate."""
    nodes = []
    for gov, (prefix, mean_t) in GOVERNORATES.items():
        for i in range(1, NODES_PER_GOVERNORATE + 1):
            nodes.append((f"{prefix}-{CROP}-{i:03d}", gov, mean_t))
    return nodes


def build_time_axis():
    """UTC time axis, one point every 10 minutes (tz-naive UTC for computations)."""
    start = pd.Timestamp(START_TS).tz_convert("UTC").tz_localize(None)
    return pd.date_range(start=start, periods=N_STEPS, freq=f"{STEP_MINUTES}min")


def format_ts(index):
    """ISO 8601 UTC strings such as 2026-11-10T00:00:00Z."""
    return index.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# 2. NORMAL SIGNALS
# =============================================================================
def ar1(rng, n_nodes, n_steps, phi, stationary_sd):
    """AR(1) process, vectorised across nodes, stationary std = stationary_sd."""
    innov_sd = stationary_sd * np.sqrt(1.0 - phi ** 2)
    x = np.empty((n_nodes, n_steps))
    x[:, 0] = rng.normal(0.0, stationary_sd, n_nodes)
    eps = rng.normal(0.0, innov_sd, (n_nodes, n_steps))
    for t in range(1, n_steps):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


def pump_mask(hours_float):
    """Boolean array: True while the pump is scheduled ON (UTC hours)."""
    on = np.zeros_like(hours_float, dtype=bool)
    for start, end in PUMP_WINDOWS_HOURS:
        on |= (hours_float >= start) & (hours_float < end)
    return on


def smooth_exponential(signal, tau):
    """First-order low-pass filter along the last axis (irrigation -> soil moisture lag)."""
    alpha = 1.0 / tau
    out = np.empty_like(signal, dtype=float)
    out[..., 0] = signal[..., 0]
    for t in range(1, signal.shape[-1]):
        out[..., t] = out[..., t - 1] + alpha * (signal[..., t] - out[..., t - 1])
    return out


def simulate_normal(rng, nodes, index):
    """Return (humidity, temperature, flow, pump_on_1d) arrays of shape (n_nodes, N_STEPS)."""
    n_nodes = len(nodes)
    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    pump_on = pump_mask(hours)                                    # shape (N_STEPS,)

    # --- soil humidity: baseline + small diurnal swing + slow drift + irrigation lag + noise
    node_offset = rng.normal(0.0, HUMIDITY_NODE_SD, (n_nodes, 1))
    diurnal_h = HUMIDITY_DIURNAL_AMP * np.sin(2 * np.pi * (hours - 3.0) / 24.0)
    drift = ar1(rng, n_nodes, N_STEPS, HUMIDITY_DRIFT_PHI, HUMIDITY_DRIFT_SD)
    irrigation = smooth_exponential(
        np.tile(pump_on.astype(float), (n_nodes, 1)), HUMIDITY_IRRIGATION_TAU
    )
    humidity = (HUMIDITY_BASE + node_offset + diurnal_h + drift
                + HUMIDITY_IRRIGATION_GAIN * (irrigation - irrigation.mean())
                + rng.normal(0.0, HUMIDITY_NOISE_SD, (n_nodes, N_STEPS)))

    # --- temperature: governorate mean + diurnal cycle (max ~15:00) + slow weather + noise
    mean_t = np.array([n[2] for n in nodes])[:, None]
    diurnal_t = TEMP_DIURNAL_AMP * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
    weather = ar1(rng, n_nodes, N_STEPS, TEMP_WEATHER_PHI, TEMP_WEATHER_SD)
    temperature = (mean_t + diurnal_t + weather
                   + rng.normal(0.0, TEMP_NOISE_SD, (n_nodes, N_STEPS)))

    # --- water flow: repeating pump-like ON/OFF cycle
    on_level = rng.uniform(*FLOW_ON_LEVEL, (n_nodes, 1))
    flow_on = on_level + rng.normal(0.0, FLOW_ON_NOISE_SD, (n_nodes, N_STEPS))
    flow_off = FLOW_OFF_LEVEL + np.abs(rng.normal(0.0, FLOW_OFF_NOISE_SD, (n_nodes, N_STEPS)))
    flow = np.where(pump_on[None, :], flow_on, flow_off)

    return humidity, temperature, flow, pump_on, hours


# =============================================================================
# 3. ANOMALY INJECTION
# =============================================================================
def episode_is_allowed(kind, start, length, pump_on, hours):
    """Scheduling constraint so that each anomaly is physically meaningful and visible."""
    sl = slice(start, start + length)
    if kind == "flow_drop":                 # needs an expected flow to drop: pump must be ON
        return bool(pump_on[sl].all())
    if kind == "unauthorized_pumping":      # pumping while the pump is scheduled OFF, at night
        h = hours[sl]
        return bool((~pump_on[sl]).all() and ((h >= 21.0) | (h < 5.0)).all())
    if kind == "sirocco_heat":              # daytime heat event
        h = hours[sl]
        return bool(((h >= 9.0) & (h < 17.0)).all())
    return True                             # low_soil_humidity, high_flow_leak: any time


def inject_anomalies(rng, humidity, temperature, flow, pump_on, hours):
    """
    Inject non-overlapping anomaly episodes in place. Returns (is_anomaly, anomaly_type) arrays.
    Each node receives ~ANOMALY_TARGET_RATE of anomalous rows, with the five types balanced.
    """
    n_nodes, n_steps = humidity.shape
    is_anom = np.zeros((n_nodes, n_steps), dtype=np.int8)
    a_type = np.full((n_nodes, n_steps), "none", dtype=object)
    occupied = np.zeros((n_nodes, n_steps), dtype=bool)
    target = int(round(ANOMALY_TARGET_RATE * n_steps))

    for n in range(n_nodes):
        count, k = 0, 0
        while count < target:
            kind = ANOMALY_TYPES[k % len(ANOMALY_TYPES)]      # round-robin => balanced types
            k += 1
            length = int(rng.integers(EPISODE_LEN[0], EPISODE_LEN[1] + 1))
            # rejection sampling of a valid, free start position
            for _ in range(10_000):
                start = int(rng.integers(0, n_steps - length))
                lo, hi = max(0, start - MIN_GAP_STEPS), min(n_steps, start + length + MIN_GAP_STEPS)
                if occupied[n, lo:hi].any():
                    continue
                if episode_is_allowed(kind, start, length, pump_on, hours):
                    break
            else:
                raise RuntimeError("Could not place an anomaly episode")
            sl = slice(start, start + length)
            noise = lambda sd: rng.normal(0.0, sd, length)

            if kind == "low_soil_humidity":          # sudden humidity drop
                humidity[n, sl] -= rng.uniform(15.0, 28.0) + noise(0.8)
            elif kind == "high_flow_leak":           # abnormally high flow
                flow[n, sl] += rng.uniform(70.0, 120.0) + noise(5.0)
            elif kind == "unauthorized_pumping":     # high flow + humidity decrease
                flow[n, sl] += rng.uniform(80.0, 130.0) + noise(5.0)
                humidity[n, sl] -= rng.uniform(8.0, 16.0) + noise(0.8)
            elif kind == "sirocco_heat":             # heat spike + humidity decrease
                temperature[n, sl] += rng.uniform(9.0, 15.0) + noise(0.6)
                humidity[n, sl] -= rng.uniform(8.0, 15.0) + noise(0.8)
            elif kind == "flow_drop":                # unexpected flow decrease
                flow[n, sl] *= rng.uniform(0.0, 0.25) + np.abs(noise(0.02))

            is_anom[n, sl] = 1
            a_type[n, sl] = kind
            occupied[n, sl] = True
            count += length
    return is_anom, a_type


# =============================================================================
# 4. DATASET ASSEMBLY
# =============================================================================
def build_main_dataset(seed):
    rng = np.random.default_rng(seed)
    nodes = build_nodes()
    index = build_time_axis()
    humidity, temperature, flow, pump_on, hours = simulate_normal(rng, nodes, index)
    is_anom, a_type = inject_anomalies(rng, humidity, temperature, flow, pump_on, hours)

    # Physical limits (hard clip) then rounding to 2 decimals
    humidity = np.round(np.clip(humidity, *LIMITS["soil_humidity"]), 2)
    temperature = np.round(np.clip(temperature, *LIMITS["temperature"]), 2)
    flow = np.round(np.clip(flow, *LIMITS["water_flow"]), 2)

    n_nodes = len(nodes)
    ts = np.tile(np.asarray(format_ts(index)), n_nodes)
    df = pd.DataFrame({
        "sensor_id": np.repeat([n[0] for n in nodes], N_STEPS),
        "soil_humidity": humidity.reshape(-1),
        "temperature": temperature.reshape(-1),
        "water_flow": flow.reshape(-1),
        "timestamp": ts,
        "governorate": np.repeat([n[1] for n in nodes], N_STEPS),
        "is_anomaly": is_anom.reshape(-1).astype(int),
        "anomaly_type": a_type.reshape(-1),
    })
    return df[MAIN_COLUMNS]


def build_grpc_dataset(main_df):
    """
    Long format: each main row -> 3 records (MOISTURE, FLOW, TEMP), in this order.
    Field names and metric_type values come from protos/tunisovereign.proto and
    TelemetryReading.MetricType. signature_hmac is left EMPTY on purpose.
    """
    n = len(main_df)
    values = np.column_stack([main_df[col].to_numpy() for _, col in METRIC_ORDER]).reshape(-1)
    grpc_df = pd.DataFrame({
        "node_code": np.repeat(main_df["sensor_id"].to_numpy(), 3),
        "governorate": np.repeat(main_df["governorate"].to_numpy(), 3),
        "metric_type": np.tile([m for m, _ in METRIC_ORDER], n),
        "value": values,
        "iso_timestamp": np.repeat(main_df["timestamp"].to_numpy(), 3),
        "signature_hmac": "",
    })
    return grpc_df[GRPC_COLUMNS]


def build_json_example(grpc_df):
    """One TelemetryPacket-shaped example taken from the first gRPC record."""
    row = grpc_df.iloc[0]
    return {
        "node_code": row["node_code"],
        "governorate": row["governorate"],
        "metric_type": row["metric_type"],
        "value": float(row["value"]),
        "iso_timestamp": row["iso_timestamp"],
        # NOT a real signature: the client must compute the HMAC at runtime with
        # the team's agreed secret and signing convention. No secret lives here.
        "signature_hmac": "<computed-at-runtime>",
    }


# =============================================================================
# 5. VALIDATION
# =============================================================================
def validate(main_df, grpc_df, example):
    """Run all checks and return a list of (description, passed, detail)."""
    res = []

    def check(desc, ok, detail=""):
        res.append((desc, bool(ok), detail))

    expected_rows = len(GOVERNORATES) * NODES_PER_GOVERNORATE * N_DAYS * 24 * 6
    check("Main dataset has 151,200 rows x 8 columns",
          main_df.shape == (expected_rows, 8) and expected_rows == 151_200, str(main_df.shape))
    check("Main columns are exactly the 8 expected, in order",
          list(main_df.columns) == MAIN_COLUMNS)
    check("gRPC dataset has 453,600 rows x 6 columns", grpc_df.shape == (453_600, 6), str(grpc_df.shape))
    check("35 unique nodes", main_df["sensor_id"].nunique() == 35, str(main_df["sensor_id"].nunique()))
    check("7 governorates", main_df["governorate"].nunique() == 7, str(main_df["governorate"].nunique()))
    per_gov = main_df.groupby("governorate")["sensor_id"].nunique()
    check("Every governorate has exactly 5 nodes", (per_gov == 5).all() and len(per_gov) == 7)
    check("Every node belongs to exactly one governorate",
          (main_df.groupby("sensor_id")["governorate"].nunique() == 1).all())
    check("No missing values in the main dataset", int(main_df.isna().sum().sum()) == 0)
    check("No duplicate (sensor_id, timestamp) pairs",
          not main_df.duplicated(["sensor_id", "timestamp"]).any())
    check("Every node has exactly 4,320 timestamps",
          (main_df.groupby("sensor_id").size() == 4320).all())

    t = pd.to_datetime(main_df["timestamp"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
    steps = t.groupby(main_df["sensor_id"]).diff().dropna()
    check("Timestamps are exactly 10 minutes apart for each node", (steps == pd.Timedelta(minutes=10)).all())
    check("Timestamps are ISO 8601 UTC ('Z'), start 2026-11-10T00:00:00Z",
          main_df["timestamp"].str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z").all()
          and main_df["timestamp"].min() == "2026-11-10T00:00:00Z")

    for col, (lo, hi) in LIMITS.items():
        check(f"{col} within documented limits [{lo}, {hi}]", main_df[col].between(lo, hi).all(),
              f"min={main_df[col].min()}, max={main_df[col].max()}")

    check("is_anomaly is only 0 or 1", set(main_df["is_anomaly"].unique()) <= {0, 1})
    check("is_anomaly = 0 exactly when anomaly_type = none",
          ((main_df["is_anomaly"] == 0) == (main_df["anomaly_type"] == "none")).all())
    check("Every anomaly label is one of the 5 allowed categories",
          set(main_df.loc[main_df["is_anomaly"] == 1, "anomaly_type"]) == set(ANOMALY_TYPES))

    # Label consistency: each anomaly must be visible in the signals it is supposed to alter
    normal = main_df[main_df["is_anomaly"] == 0]
    h_lo, h_hi = NORMAL_HUMIDITY_EXPECTED
    share = normal["soil_humidity"].between(h_lo, h_hi).mean()
    check("Normal humidity generally within 45-72 % (>= 99.5 % of normal rows)", share >= 0.995,
          f"{share:.4%}")
    q = normal[["soil_humidity", "temperature", "water_flow"]]
    a = main_df[main_df["is_anomaly"] == 1]
    flow_hi = normal["water_flow"].max()
    check("low_soil_humidity rows have humidity below normal 1st percentile",
          (a[a.anomaly_type == "low_soil_humidity"]["soil_humidity"] < q["soil_humidity"].quantile(0.01)).mean() > 0.9)
    check("high_flow_leak / unauthorized_pumping rows show flow above scheduled level",
          (a[a.anomaly_type == "high_flow_leak"]["water_flow"] > 60).all()
          and (a[a.anomaly_type == "unauthorized_pumping"]["water_flow"] > 60).all())
    check("sirocco_heat rows have temperature above normal 99th percentile",
          (a[a.anomaly_type == "sirocco_heat"]["temperature"] > q["temperature"].quantile(0.99)).mean() > 0.9)
    check("flow_drop rows have flow below 50 L/min while pump is scheduled ON",
          (a[a.anomaly_type == "flow_drop"]["water_flow"] < 50).all())

    # gRPC checks
    check("Each main row produces exactly 3 gRPC records", len(grpc_df) == 3 * len(main_df))
    check("gRPC metric types are exactly MOISTURE, FLOW, TEMP",
          set(grpc_df["metric_type"].unique()) == {"MOISTURE", "FLOW", "TEMP"})
    check("Each (node, timestamp) has one MOISTURE, one FLOW and one TEMP record",
          (grpc_df.groupby(["node_code", "iso_timestamp"])["metric_type"].nunique() == 3).all())
    check("Every gRPC node_code matches a main sensor_id",
          set(grpc_df["node_code"]) == set(main_df["sensor_id"]))
    check("All node_code values fit ResourceNodePlot.node_code (max_length=32)",
          grpc_df["node_code"].str.len().max() <= 32)
    # value round-trip against the main dataset
    back = grpc_df["value"].to_numpy().reshape(-1, 3)
    orig = main_df[["soil_humidity", "water_flow", "temperature"]].to_numpy()
    check("gRPC values equal main values (MOISTURE/FLOW/TEMP mapping)", np.array_equal(back, orig))
    check("HMAC fields are intentionally empty in the gRPC CSV", (grpc_df["signature_hmac"] == "").all())
    check("JSON example uses exactly the 6 TelemetryPacket fields and a placeholder HMAC",
          list(example.keys()) == GRPC_COLUMNS and example["signature_hmac"] == "<computed-at-runtime>")
    return res


def print_report(results):
    print("\n" + "=" * 78)
    print("VALIDATION REPORT")
    print("=" * 78)
    for desc, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}" + (f"  ({detail})" if detail else ""))
    n_ok = sum(ok for _, ok, _ in results)
    print("-" * 78)
    print(f"{n_ok}/{len(results)} checks passed")
    return n_ok == len(results)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_df = build_main_dataset(args.seed)
    grpc_df = build_grpc_dataset(main_df)
    example = build_json_example(grpc_df)

    # Fixed float format + '\n' line endings => byte-identical files across runs/platforms
    main_df.to_csv(args.out_dir / MAIN_CSV, index=False, float_format="%.2f", lineterminator="\n")
    grpc_df.to_csv(args.out_dir / GRPC_CSV, index=False, float_format="%.2f", lineterminator="\n")
    with open(args.out_dir / JSON_EXAMPLE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Seed: {args.seed}")
    print(f"Main dataset : {main_df.shape}  -> {args.out_dir / MAIN_CSV}")
    print(f"gRPC dataset : {grpc_df.shape}  -> {args.out_dir / GRPC_CSV}")
    print(f"JSON example : {args.out_dir / JSON_EXAMPLE}")
    rate = main_df["is_anomaly"].mean()
    print(f"Anomaly rate : {rate:.2%}")
    print(main_df[main_df.is_anomaly == 1]["anomaly_type"].value_counts().to_string())

    # Validate the data that was actually written to disk (re-read the CSVs)
    main_disk = pd.read_csv(args.out_dir / MAIN_CSV, keep_default_na=False)
    grpc_disk = pd.read_csv(args.out_dir / GRPC_CSV, keep_default_na=False)
    with open(args.out_dir / JSON_EXAMPLE, encoding="utf-8") as f:
        example_disk = json.load(f)
    ok = print_report(validate(main_disk, grpc_disk, example_disk))

    print("\nSHA-256 (reproducibility check):")
    for name in (MAIN_CSV, GRPC_CSV, JSON_EXAMPLE):
        print(f"  {sha256_of(args.out_dir / name)}  {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
