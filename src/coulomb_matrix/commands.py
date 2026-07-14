"""Console entry point implementations for the coulomb_matrix package.

Provides: `xsf-to-npy`, `coulomb-matrix`, `coulomb-exchange` console scripts.
"""
import argparse
import glob
import os
import re
import time

import gpaw.mpi as mpi
import numpy as np

from .config import CoulombConfig
from .eri_utils import uniquify
from .j_matrix import JCoulombCalculator
from .v_matrix import VCoulombCalculator
from .xsf import Xsf2Np


def xsf_to_npy(argv=None):
    """Entry point for converting .xsf files to .npy"""
    parser = argparse.ArgumentParser(
        description="Convert .xsf Wannier functions to .npy files"
    )
    parser.add_argument(
        "--pattern",
        default="*.xsf",
        help="Glob pattern to find xsf files (default '*.xsf')",
    )
    parser.add_argument(
        "--out-prefix", default=None, help="Prefix for output npy filenames"
    )
    parser.add_argument("--out-dir", default=None, help="Directory to write .npy files")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .npy files"
    )
    args = parser.parse_args(argv)

    def find_files(pattern):
        files = glob.glob(pattern)
        files.sort(
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else x
        )
        return files

    def convert_file(path, out_prefix=None, out_dir=None, overwrite=False):
        print(f"Processing file: {path}", flush=True)
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
            print(f"Skip existing: {out_path}", flush=True)
            return out_path
        np.save(out_path, wf)
        return out_path

    files = find_files(args.pattern)
    if not files:
        print("No files found for pattern:", args.pattern)
        return 1
    for p in files:
        try:
            out = convert_file(
                p,
                out_prefix=args.out_prefix,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
            )
            print("Wrote:", out)
        except Exception as e:
            print(f"Failed {p}: {e}")
    return 0


def coulomb_matrix(argv=None):
    """Run the packaged create_coulomb_matrix implementation."""
    os.environ["OMP_NUM_THREADS"] = "1"  # GPAW warning suggests OMP_NUM_THREADS=1
    parser = argparse.ArgumentParser(
        description="Compute the Coulomb matrix (V) from Wannier functions."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to TOML config file."
    )
    parser.add_argument(
        "--help-config", 
        action="store_true", 
        help="Show detailed help for all TOML configuration parameters and exit."
    )
    args = parser.parse_args(argv)

    # Intercept early if config help is requested before any MPI/Calculator overhead
    if args.help_config:
        if mpi.world.rank == 0:
            print(CoulombConfig.get_help_text())
        return 0

    # Time calculation
    if mpi.world.rank == 0:
        start_time = time.time()
        print("Starting calculation of Coulomb matrix...", flush=True)
        # Time the initialization
        init_start_time = time.time()
        print("Initializing VCoulombCalculator...", flush=True)
    calc = VCoulombCalculator(config_path=args.config)
    mpi.world.barrier()  # Ensure all processes reach this point before timing
    if mpi.world.rank == 0:
        print(
            f"VCoulombCalculator initialized in {time.time() - init_start_time:.2f} seconds",
            flush=True,
        )
        # Time the run
        print("Running VCoulombCalculator...", flush=True)
        run_start_time = time.time()
    V = calc.run()
    mpi.world.barrier()  # Ensure all processes finish before timing
    if mpi.world.rank == 0:
        print(
            f"VCoulombCalculator ran in {time.time() - run_start_time:.2f} seconds",
            flush=True,
        )
        # Time the save
        print("Saving results...", flush=True)
        save_start_time = time.time()

        output_cfg = calc.config.output
        save_dir = output_cfg.save_dir
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        filename = output_cfg.matrix_filename
        path = os.path.join(save_dir, filename)
        if output_cfg.use_unique_filenames:
            path = uniquify(path)
        np.save(path, V)
        print(
            f"Results saved in {time.time() - save_start_time:.2f} seconds", flush=True
        )
        print(
            f"Total time for Coulomb matrix calculation: {time.time() - start_time:.2f} seconds",
            flush=True,
        )
    return 0


def coulomb_exchange(argv=None):
    """Run the packaged create_exchange_matrix implementation."""
    os.environ["OMP_NUM_THREADS"] = "1"  # GPAW warning suggests OMP_NUM_THREADS=1
    parser = argparse.ArgumentParser(description="Compute J Coulomb matrix")
    # TODO: TOML should not be optional.
    parser.add_argument("--config", default=None, help="Optional TOML config file")
    parser.add_argument(
        "--help-config", 
        action="store_true", 
        help="Show detailed help for all TOML configuration parameters and exit."
    )
    args = parser.parse_args(argv)

    # Intercept early
    if args.help_config:
        if mpi.world.rank == 0:
            print(CoulombConfig.get_help_text())
        return 0

    # Time calculation
    if mpi.world.rank == 0:
        start_time = time.time()
        print("Starting calculation of exchange Coulomb matrix...", flush=True)
        # Time the initialization
        init_start_time = time.time()
        print("Initializing JCoulombCalculator...", flush=True)
    calc = JCoulombCalculator(config_path=args.config)
    mpi.world.barrier()  # Ensure all processes reach this point before timing
    if mpi.world.rank == 0:
        print(
            f"JCoulombCalculator initialized in {time.time() - init_start_time:.2f} seconds",
            flush=True,
        )
        # Time the run
        print("Running JCoulombCalculator...", flush=True)
        run_start_time = time.time()
    V = calc.run()
    mpi.world.barrier()  # Ensure all processes finish before timing
    if mpi.world.rank == 0:
        print(
            f"JCoulombCalculator ran in {time.time() - run_start_time:.2f} seconds",
            flush=True,
        )
        # Time the save
        print("Saving results...", flush=True)
        save_start_time = time.time()

        output_cfg = calc.config.output
        save_dir = output_cfg.save_dir
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        filename = output_cfg.exchange_matrix_filename
        path = os.path.join(save_dir, filename)
        if output_cfg.use_unique_filenames:
            path = uniquify(path)
        np.save(path, V)
        print(
            f"Results saved in {time.time() - save_start_time:.2f} seconds", flush=True
        )
        print(
            f"Total time for exchange Coulomb matrix calculation: {time.time() - start_time:.2f} seconds",
            flush=True,
        )
    return 0
