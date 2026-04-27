"""
Data processing pipeline for tool wear prediction.

Reads raw force/torque (.txt), vibration/sound (.csv/.xlsx), and tool wear (.xls),
extracts statistical features from the time-series data using a sliding window
approach, averages Vbmax per cycle, and saves an 80/20 train/test split.
"""

import os
import re
import random
import pathlib
import zipfile
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

# ─── paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "raw_data" / "Milling dataset from QIT"
TOOL_WEAR_PATH = RAW_DIR / "tool wear.xls"
FORCE_DIR = RAW_DIR / "Force and torque data"
VIB_DIR = RAW_DIR / "Vibration and sound data"
OUT_DIR = PROJECT_ROOT / "processed_data"

RANDOM_SEED = 42
NUM_WINDOWS = 20


# ─── helpers ─────────────────────────────────────────────────────────────────
def normalize_id(filename: str) -> str:
    """Extract a canonical 'MM-DD-N' id from a filename, stripping extensions
    and leading zeros on the run number so that '01-26-01' == '01-26-1'."""
    stem = pathlib.Path(filename).stem
    m = re.match(r"(\d{2})-(\d{2})-0*(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse date-run id from '{filename}'")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def extract_time_domain_features(signal: np.ndarray) -> dict:
    """Compute time-domain statistical features for a 1-D signal."""
    return {
        "mean": np.mean(signal),
        "std": np.std(signal, ddof=1) if len(signal) > 1 else 0.0,
        "max": np.max(signal),
        "min": np.min(signal),
        "kurtosis": float(stats.kurtosis(signal, fisher=True)),
        "skewness": float(stats.skew(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "p25": float(np.percentile(signal, 25)),
        "p75": float(np.percentile(signal, 75)),
    }


def extract_mean_frequency(signal: np.ndarray, fs: float = 10000.0) -> dict:
    """Compute the mean (weighted-average) frequency via FFT."""
    n = len(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(n, d=1.0 / fs)

    yf = yf[1:]
    xf = xf[1:]

    if len(yf) == 0:
        return {"mean_freq": 0.0}

    yf_sum = np.sum(yf)
    mean_freq = float(np.sum(xf * yf) / yf_sum) if yf_sum > 0 else 0.0

    return {"mean_freq": mean_freq}


def extract_channel_features(signal: np.ndarray, channel_name: str) -> dict:
    """Full feature set (time-domain + mean frequency) for one signal channel."""
    td = extract_time_domain_features(signal)
    fd = extract_mean_frequency(signal)
    return {f"{channel_name}_{k}": v for k, v in {**td, **fd}.items()}


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx_numeric_columns(path: pathlib.Path) -> np.ndarray:
    """Stream-parse a large .xlsx and return columns B-E as a float array.

    Bypasses the shared-strings table entirely (we only need numeric cells).
    Returns shape (n_rows, 4) for the 4 sensor channels.
    Tolerates truncated/corrupted XML by returning data collected so far."""
    rows: list[list[float]] = []
    with zipfile.ZipFile(path) as zf:
        with zf.open("xl/worksheets/sheet.xml") as f:
            try:
                for event, elem in iterparse(f, events=("end",)):
                    if elem.tag != f"{NS}row":
                        continue
                    row_num = int(elem.get("r", "0"))
                    if row_num <= 1:
                        elem.clear()
                        continue
                    vals = [np.nan] * 4
                    for c in elem:
                        if c.tag != f"{NS}c":
                            continue
                        ref = c.get("r", "")
                        if not ref or ref[0] not in "BCDE":
                            continue
                        if c.get("t") == "s":
                            continue
                        v_elem = c.find(f"{NS}v")
                        if v_elem is not None and v_elem.text:
                            col_idx = ord(ref[0]) - ord("B")
                            vals[col_idx] = float(v_elem.text)
                    rows.append(vals)
                    elem.clear()
            except Exception as e:
                print(f"    WARNING: XML parse error in {path.name} at row "
                      f"{len(rows)}: {e}. Using {len(rows)} rows collected.")
    arr = np.array(rows, dtype=np.float64)
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask]


# ─── step 1: load tool wear labels ──────────────────────────────────────────
def load_tool_wear() -> pd.Series:
    """Return a Series mapping cycle number (int) -> average Vbmax (float)."""
    df = pd.read_excel(TOOL_WEAR_PATH, engine="xlrd", header=None)

    vbmax_cols = [1, 4, 7, 10]
    data = df.iloc[4:].copy()
    data.columns = range(data.shape[1])

    cycles = data[0].astype(int)
    vbmax_values = data[vbmax_cols].astype(float)
    avg_vbmax = vbmax_values.mean(axis=1)

    result = pd.Series(avg_vbmax.values, index=cycles.values, name="avg_vbmax")
    print(f"  Loaded tool wear data: {len(result)} cycles, "
          f"Vbmax range [{result.min():.4f}, {result.max():.4f}]")
    return result


# ─── step 2: match files to cycles ──────────────────────────────────────────
def build_cycle_map() -> list[dict]:
    """Return a list of dicts sorted by cycle order, each containing:
    cycle, force_file, vib_file."""
    force_files = {normalize_id(f): f for f in os.listdir(FORCE_DIR)}
    vib_files = {normalize_id(f): f for f in os.listdir(VIB_DIR)}

    common_ids = sorted(set(force_files) & set(vib_files),
                        key=lambda x: tuple(int(p) for p in x.split("-")))

    cycle_map = []
    for cycle_num, fid in enumerate(common_ids, start=1):
        cycle_map.append({
            "cycle": cycle_num,
            "id": fid,
            "force_file": force_files[fid],
            "vib_file": vib_files[fid],
        })

    print(f"  Matched {len(cycle_map)} cycles (intersection of both datasets)")
    return cycle_map


# ─── step 3: load raw signals for one cycle ─────────────────────────────────
def load_raw_signals(entry: dict) -> dict[str, np.ndarray] | None:
    """Return a dict of channel_name -> 1-D numpy array for one cycle,
    or None if a file is empty/corrupt."""
    signals = {}

    ft_path = FORCE_DIR / entry["force_file"]
    if ft_path.stat().st_size == 0:
        return None
    ft_df = pd.read_csv(ft_path, sep="\t")
    for col in ["Fx", "Fy", "Fz", "Mz"]:
        signals[col] = ft_df[col].values

    vib_path = VIB_DIR / entry["vib_file"]
    if vib_path.stat().st_size == 0:
        return None
    channel_names = ["accel_x", "accel_y", "accel_z", "sound"]
    if vib_path.suffix == ".xlsx":
        vib_data = read_xlsx_numeric_columns(vib_path)
        for col_idx, name in enumerate(channel_names):
            signals[name] = vib_data[:, col_idx]
    else:
        vib_df = pd.read_csv(vib_path, encoding="latin-1", on_bad_lines="skip")
        for col_idx, name in enumerate(channel_names):
            col_data = pd.to_numeric(vib_df.iloc[:, col_idx + 1], errors="coerce")
            signals[name] = col_data.dropna().values

    return signals


def detect_cutting_region(signals: dict[str, np.ndarray]) -> tuple[int, int]:
    """Detect the active cutting region using Fz (axial force).

    Computes a 1-second rolling RMS of Fz and marks regions where
    it exceeds 5% of the peak RMS as active cutting."""
    fz = signals["Fz"]
    window = min(10000, len(fz) // 4)
    rms = np.sqrt(np.convolve(fz ** 2, np.ones(window) / window, mode="same"))
    threshold = rms.max() * 0.05
    active = np.where(rms > threshold)[0]
    if len(active) == 0:
        return 0, len(fz)
    return int(active[0]), int(active[-1]) + 1


def extract_windowed_features(signals: dict[str, np.ndarray],
                              num_windows: int) -> list[dict]:
    """Trim to the active cutting region, then split into num_windows equal
    segments and extract features from each window."""
    cut_start, cut_end = detect_cutting_region(signals)
    trimmed = {k: v[cut_start:cut_end] for k, v in signals.items()}

    min_len = min(len(s) for s in trimmed.values())
    window_size = min_len // num_windows
    if window_size < 100:
        num_windows = max(1, min_len // 100)
        window_size = min_len // num_windows

    window_rows = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        features = {"window": w}
        for ch_name, full_signal in trimmed.items():
            seg = full_signal[start:end]
            features.update(extract_channel_features(seg, ch_name))
        window_rows.append(features)

    return window_rows


# ─── step 4 & 5: build dataset, split, and save ─────────────────────────────
def main():
    print("Step 1: Loading tool wear labels...")
    vbmax = load_tool_wear()

    print("Step 2: Building file-to-cycle map...")
    cycle_map = build_cycle_map()

    print(f"Step 3: Extracting windowed features ({NUM_WINDOWS} windows/cycle)...")
    all_rows = []
    for i, entry in enumerate(cycle_map):
        cycle = entry["cycle"]
        if cycle not in vbmax.index:
            print(f"  WARNING: cycle {cycle} not found in tool wear data, skipping")
            continue

        signals = load_raw_signals(entry)
        if signals is None:
            print(f"  [{i + 1}/{len(cycle_map)}] Cycle {cycle} ({entry['id']}) "
                  f"-> SKIPPED (empty file)")
            continue

        cut_start, cut_end = detect_cutting_region(signals)
        total_len = len(signals["Fz"])
        cut_pct = (cut_end - cut_start) / total_len * 100

        window_rows = extract_windowed_features(signals, NUM_WINDOWS)

        for row in window_rows:
            row["cycle"] = cycle
            row["avg_vbmax"] = vbmax[cycle]
            all_rows.append(row)

        print(f"  [{i + 1}/{len(cycle_map)}] Cycle {cycle} ({entry['id']}) "
              f"-> {len(window_rows)} windows "
              f"(cutting: {cut_pct:.0f}%, trimmed {cut_start/10000:.1f}s idle)")

    df = pd.DataFrame(all_rows)
    feature_cols = [c for c in df.columns
                    if c not in ("cycle", "avg_vbmax", "window")]
    print(f"\n  Total samples: {len(df)}, features per sample: {len(feature_cols)}")

    # ── 80/20 split at the CYCLE level to prevent data leakage ──
    random.seed(RANDOM_SEED)
    unique_cycles = df["cycle"].unique().tolist()
    random.shuffle(unique_cycles)
    split = int(0.8 * len(unique_cycles))
    train_cycles = set(unique_cycles[:split])
    test_cycles = set(unique_cycles[split:])

    train_df = df[df["cycle"].isin(train_cycles)].reset_index(drop=True)
    test_df = df[df["cycle"].isin(test_cycles)].reset_index(drop=True)

    # ── save ──
    train_dir = OUT_DIR / "train"
    test_dir = OUT_DIR / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_df[feature_cols].to_csv(train_dir / "train_features.csv", index=False)
    train_df[["avg_vbmax"]].to_csv(train_dir / "train_labels.csv", index=False)
    test_df[feature_cols].to_csv(test_dir / "test_features.csv", index=False)
    test_df[["avg_vbmax"]].to_csv(test_dir / "test_labels.csv", index=False)

    print(f"\nDone!  Train: {len(train_df)} samples ({len(train_cycles)} cycles)  "
          f"|  Test: {len(test_df)} samples ({len(test_cycles)} cycles)")
    print(f"  Saved to {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
