"""Console entry point implementations for the coulomb_matrix package.

Provides: `xsf-to-npy`, `coulomb-matrix`, `coulomb-exchange` console scripts.
"""
import os
import argparse
import glob
import re
import numpy as np
from .xsf import Xsf2Np
from .v_matrix import VCoulombCalculator
from .j_matrix import JCoulombCalculator
import gpaw.mpi as mpi
from .eri_utils import uniquify


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


def coulomb_matrix(argv=None):
    """Run the packaged create_coulomb_matrix implementation."""
    parser = argparse.ArgumentParser(description="Compute the Coulomb matrix (V) from Wannier functions.")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file.")
    args = parser.parse_args(argv)
    calc = VCoulombCalculator(config_path=args.config)
    V = calc.run()

    # save by rank 0
    if not mpi.world.rank:
        output_cfg = calc.config.get("output", {})
        save_dir = output_cfg.get("save_dir", "./")
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        filename = output_cfg.get("matrix_filename", "V_matrix.npy" if calc.mode == "V" else "J_matrix.npy")
        path = os.path.join(save_dir, filename)
        if output_cfg.get("use_unique_filenames", True):
            path = uniquify(path)
        np.save(path, V)
    return 0


def coulomb_exchange(argv=None):
    """Run the packaged create_exchange_matrix implementation."""
    parser = argparse.ArgumentParser(description="Compute J Coulomb matrix")
    parser.add_argument("--xsf-dir", default="./", help="Directory for xsf files")
    parser.add_argument("--npy-dir", default="./", help="Directory for npy files")
    parser.add_argument("--config", default=None, help="Optional TOML config file")
    args = parser.parse_args(argv)
    calc = JCoulombCalculator(config_path=args.config, xsf_dir=args.xsf_dir, npy_dir=args.npy_dir)
    V = calc.run()

    # save by rank 0
    if not mpi.world.rank:
        output_cfg = calc.config.get("output", {})
        save_dir = output_cfg.get("save_dir", "./")
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        filename = output_cfg.get("matrix_filename", "V_matrix.npy" if calc.mode == "V" else "J_matrix.npy")
        path = os.path.join(save_dir, filename)
        if output_cfg.get("use_unique_filenames", True):
            path = uniquify(path)
        np.save(path, V)
    return 0

