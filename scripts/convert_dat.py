#!/usr/bin/env python3
"""
Converts NASA PSI raw .dat field-dump files into readable CSVs.

Searches the input folder recursively. Output folder is flat (one .csv per
.dat, filenames are assumed unique) plus a manifest.csv summarizing what was
found in each file, including its original relative path.

Handles two known layouts:
  1. Files with a 4-int header line (nx ny nz nvars) followed by nx*ny
     whitespace/newline-separated floats -> reshaped into an (ny, nx) grid.
  2. Files with no valid header (or where nx*ny doesn't match the data
     count) -> saved as a flat single-column CSV, no reshaping attempted.

Usage:
    python convert_dat.py <input_dir> <output_dir>
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

def try_parse_header(first_line):
    """Return (nx, ny, nz, nvars) if the first line looks like a 4-int header, else None."""
    parts = first_line.split()
    if len(parts) == 4:
        try:
            ints = [int(p) for p in parts]
            if all(i > 0 for i in ints):
                return tuple(ints)
        except ValueError:
            pass
    return None

def convert_file(path, out_dir, in_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    result = {"file": os.path.basename(path),
              "source": os.path.relpath(path, in_dir).replace(os.sep, "/"),
              "status": "", "shape": "", "notes": ""}

    with open(path, "r") as f:
        first_line = f.readline()

    header = try_parse_header(first_line)

    # Detect a plain-text (non-numeric) header line, e.g. "Diameter (starting from...)"
    first_tokens = first_line.split()
    text_header = False
    if first_tokens:
        try:
            float(first_tokens[0])
        except ValueError:
            text_header = True

    try:
        if header:
            nx, ny, nz, nvars = header
            expected = nx * ny * nz * nvars
            # try flexible column counts (grid3d-style files pack coords oddly)
            try:
                data = np.loadtxt(path, skiprows=1).ravel()
            except ValueError:
                # ragged rows (e.g. coordinate dump with inconsistent columns) ->
                # read whitespace-split tokens manually, ignoring line structure
                with open(path) as f:
                    f.readline()
                    data = np.array([float(x) for x in f.read().split()])

            if data.size >= expected and nz == 1 and nvars == 1:
                # trim any trailing stray values (e.g. off-by-one blank line) and reshape
                grid = data[:expected].reshape(ny, nx)
                out_path = os.path.join(out_dir, f"{name}.csv")
                pd.DataFrame(grid).to_csv(out_path, index=False, header=False)
                extra = data.size - expected
                note = f"header=({nx},{ny},{nz},{nvars})"
                if extra:
                    note += f"; trimmed {extra} trailing stray value(s)"
                result.update(status="reshaped_2d", shape=f"{ny}x{nx}", notes=note)
            else:
                out_path = os.path.join(out_dir, f"{name}.csv")
                pd.DataFrame({"value": data}).to_csv(out_path, index=False)
                result.update(status="flat_fallback", shape=f"{data.size} values",
                              notes=f"header=({nx},{ny},{nz},{nvars}) did not match/exceed data size {data.size}")
        elif text_header:
            # first line is a text label, not data -> skip it, read the rest as floats
            data = np.loadtxt(path, skiprows=1).ravel()
            out_path = os.path.join(out_dir, f"{name}.csv")
            pd.DataFrame({"value": data}).to_csv(out_path, index=False)
            result.update(status="flat_text_header", shape=f"{data.size} values",
                          notes=f"skipped text header: {first_line.strip()!r}")
        else:
            data = np.loadtxt(path).ravel()
            out_path = os.path.join(out_dir, f"{name}.csv")
            pd.DataFrame({"value": data}).to_csv(out_path, index=False)
            result.update(status="flat_no_header", shape=f"{data.size} values")

    except Exception as e:
        result.update(status="FAILED", notes=str(e))

    return result

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    dat_files = sorted(glob.glob(os.path.join(in_dir, "**", "*.dat"), recursive=True))
    if not dat_files:
        print(f"No .dat files found in {in_dir}")
        return

    manifest = []
    for path in dat_files:
        print(f"Converting {os.path.basename(path)} ...", end=" ")
        r = convert_file(path, out_dir, in_dir)
        print(r["status"])
        manifest.append(r)

    pd.DataFrame(manifest).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"\nDone. {len(dat_files)} files processed.")
    print(f"Converted CSVs + manifest.csv are in: {out_dir}")

if __name__ == "__main__":
    main()