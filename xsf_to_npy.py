#!/usr/bin/env python3
"""Convert all .xsf files in the current directory to .npy Wannier function arrays.

Usage: run in the directory containing .xsf files or pass --pattern.
"""
import argparse
import glob
import re
import os
import numpy as np
from coulomb_matrix.xsf import Xsf2Np


def find_files(pattern):
    files = glob.glob(pattern)
    files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else x)
    return files


def convert_file(path, out_prefix=None, out_dir=None, overwrite=False):
    print(f"Processing file: {path}")
    with open(path, "r") as f:
        xsf = Xsf2Np(f)
        wf = xsf.get_WF()
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    if out_prefix:
        name = f"{out_prefix}_{name}"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name + ".npy")
    else:
        out_path = name + ".npy"
    if (not overwrite) and os.path.exists(out_path):
        print(f"Skip existing: {out_path}")
        return out_path
    np.save(out_path, wf)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert .xsf Wannier functions to .npy files")
    parser.add_argument("--pattern", default="*.xsf", help="Glob pattern to find xsf files (default '*.xsf')")
    parser.add_argument("--out-prefix", default=None, help="Prefix for output npy filenames")
    parser.add_argument("--out-dir", default=None, help="Directory to write .npy files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    args = parser.parse_args()

    files = find_files(args.pattern)
    if not files:
        print("No files found for pattern:", args.pattern)
        return
    for p in files:
        try:
            out = convert_file(p, out_prefix=args.out_prefix, out_dir=args.out_dir, overwrite=args.overwrite)
            print("Wrote:", out)
        except Exception as e:
            print(f"Failed {p}: {e}")


if __name__ == "__main__":
    main()
