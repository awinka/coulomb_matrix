"""Console entry point implementations for the coulomb_matrix package.

Provides: `xsf-to-npy`, `coulomb-v-ijij`, `coulomb-v-ijji` console scripts.
"""
import os
import sys
import argparse
import glob
import re
import numpy as np
from .xsf import Xsf2Np


def xsf_to_npy(argv=None):
    """Entry point for converting .xsf files to .npy"""
    parser = argparse.ArgumentParser(description="Convert .xsf Wannier functions to .npy files")
    parser.add_argument("--pattern", default="*.xsf", help="Glob pattern to find xsf files (default '*.xsf')")
    parser.add_argument("--out-prefix", default=None, help="Prefix for output npy filenames")
    parser.add_argument("--out-dir", default=None, help="Directory to write .npy files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    args = parser.parse_args(argv)

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

    files = find_files(args.pattern)
    if not files:
        print("No files found for pattern:", args.pattern)
        return 1
    for p in files:
        try:
            out = convert_file(p, out_prefix=args.out_prefix, out_dir=args.out_dir, overwrite=args.overwrite)
            print("Wrote:", out)
        except Exception as e:
            print(f"Failed {p}: {e}")
    return 0


def _run_script(script_name, argv=None):
    raise RuntimeError("_run_script is no longer supported; call the package entry points directly")


def coulomb_v_ijij(argv=None):
    """Run the packaged create_coulomb_matrix implementation."""
    from coulomb_matrix.create_coulomb_matrix import main as _main
    return _main(argv)


def coulomb_v_ijji(argv=None):
    """Run the packaged create_ijji_matrix implementation."""
    from coulomb_matrix.ijji import main as _main
    return _main(argv)


if __name__ == "__main__":
    xsf_to_npy()
