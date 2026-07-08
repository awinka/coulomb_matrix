#!/usr/bin/env python3
"""Thin wrapper to call the packaged `coulomb_matrix.ijji` implementation."""

import argparse
import os
import numpy as np
import gpaw.mpi as mpi
from coulomb_matrix.ijji import IjjiCoulombCalculator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute V_ijji Coulomb matrix")
    parser.add_argument("--xsf-dir", default="./", help="Directory for xsf files")
    parser.add_argument("--npy-dir", default="./", help="Directory for npy files")
    parser.add_argument("--config", default=None, help="Optional TOML config file")
    args = parser.parse_args(argv)
    calc = IjjiCoulombCalculator(config_path=args.config, xsf_dir=args.xsf_dir, npy_dir=args.npy_dir)
    calc.run()
    
    # save by rank 0
    if not mpi.world.rank:
        output_cfg = calc.config.get("output", {})
        save_dir = output_cfg.get("save_dir", "./")
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        filename = output_cfg.get("matrix_filename", "V_matrix.npy" if calc.mode == "ijij" else "V_matrix_ijji.npy")
        path = os.path.join(save_dir, filename)
        if output_cfg.get("use_unique_filenames", True):
            path = utils.uniquify(path)
        np.save(path, V)

if __name__ == "__main__":
    main()

