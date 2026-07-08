#!/usr/bin/env python3
"""Thin wrapper to call the packaged `coulomb_matrix.create_coulomb_matrix` implementation."""

import os
import numpy as np
import gpaw.mpi as mpi
import argparse
from coulomb_matrix.create_coulomb_matrix import CreateCoulombCalculator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the Coulomb matrix (V_ijij) from Wannier functions.")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config file.")
    args = parser.parse_args(argv)
    calc = CreateCoulombCalculator(config_path=args.config)
    V = calc.run()

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

