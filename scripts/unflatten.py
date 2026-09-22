"""Rebuild NASA PSI folder structure from flattened download zip(s).

NASA's bulk download flattens folders. Two ways to recover the tree:

1. Layout map (preferred, exact): a JSON file mapping each zip filename to
   its path on the NASA site, written by hand from the site's folder view:
       {"8cmGravityU.dat": "Analyzed Data/Modeling Data/Case1/Earth Gravity/8cmGravityU.dat"}
   Needed when the site nests more than one level deep, or when the zip
   renames files.

2. Filename prefix (fallback, top level only): NASA sometimes encodes the
   first folder level in the name as <PSI-ID>_<Folder>_<file>, e.g.
   "PSI-117_Analyzed Data_MST_2024_Dext.csv" -> "Analyzed Data/MST_2024_Dext.csv"

Files matched by neither are placed at the output root unchanged.

Usage:
    python unflatten.py <output_dir> <zip> [<zip> ...] [--layout layout.json]
"""

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

PREFIX = re.compile(r"^(PSI-\d+)_(.+?)_(.+)$")


def path_from_prefix(name: str) -> PurePosixPath:
    m = PREFIX.match(name)
    if not m:
        return PurePosixPath(name)
    return PurePosixPath(m.group(2)) / m.group(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--layout", type=Path)
    args = ap.parse_args()

    layout: dict[str, str] = {}
    if args.layout:
        layout = json.loads(args.layout.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tree: dict[str, list[str]] = defaultdict(list)
    unmapped: list[str] = []
    seen: set[str] = set()

    for zip_path in args.zips:
        with zipfile.ZipFile(zip_path) as zf:
            for entry in zf.infolist():
                if entry.is_dir():
                    continue
                name = Path(entry.filename).name
                seen.add(name)
                if name in layout:
                    rel = PurePosixPath(layout[name])
                else:
                    rel = path_from_prefix(name)
                    if layout:
                        unmapped.append(name)
                dest = args.out_dir / Path(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                tree[str(rel.parent)].append(rel.name)

    print(f"Extracted to {args.out_dir}\n")
    for folder in sorted(tree):
        print(f"{folder}/")
        for fname in sorted(tree[folder]):
            print(f"    {fname}")

    if unmapped:
        print("\nWARNING: in zip but not in layout (used prefix fallback):")
        for n in unmapped:
            print(f"    {n}")
    missing = sorted(set(layout) - seen)
    if missing:
        print("\nWARNING: in layout but not in zip:")
        for n in missing:
            print(f"    {n}")


if __name__ == "__main__":
    main()
