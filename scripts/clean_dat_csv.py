#!/usr/bin/env python3
"""
Cleans converted PSI .dat->.csv files so they're safe to load and use.

Fixes applied:
  - Replaces -inf / inf with NaN (common when a source field was stored
    in log-scale and some cells were exactly zero -> log(0) = -inf).
  - Reports how many cells were affected per file.
  - Leaves the grid shape and all finite values untouched.

Usage:
    python clean_dat_csv.py <input_dir> <output_dir>
(same filenames; manifest.csv is copied through untouched)
"""

import sys, os, glob, shutil
import numpy as np
import pandas as pd

def clean_file(path, out_dir):
    name = os.path.basename(path)
    # detect flat ("value" column) vs grid (headerless) format
    df = pd.read_csv(path)
    is_flat = list(df.columns) == ["value"]
    if not is_flat:
        df = pd.read_csv(path, header=None)

    arr = df.values.astype(float)
    n_bad = np.count_nonzero(np.isinf(arr))
    arr[np.isinf(arr)] = np.nan

    out_path = os.path.join(out_dir, name)
    if is_flat:
        pd.DataFrame({"value": arr.ravel()}).to_csv(out_path, index=False)
    else:
        pd.DataFrame(arr).to_csv(out_path, index=False, header=False)

    return name, arr.shape, n_bad, int(np.isnan(arr).sum())

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "manifest.csv"]
    if not files:
        print(f"No .csv files found in {in_dir}")
        return
    manifest = os.path.join(in_dir, "manifest.csv")
    if os.path.exists(manifest):
        shutil.copy(manifest, out_dir)

    print(f"{'file':40s} {'shape':>14s} {'inf->NaN':>10s} {'total NaN':>10s}")
    for f in files:
        name, shape, n_bad, n_nan = clean_file(f, out_dir)
        flag = "  <-- fixed" if n_bad else ""
        print(f"{name:40s} {str(shape):>14s} {n_bad:>10d} {n_nan:>10d}{flag}")

    print(f"\nDone. Cleaned files written to: {out_dir}")

if __name__ == "__main__":
    main()
