"""`ijji` Coulomb calculator subclass and CLI entrypoint."""

import argparse
import os
import numpy as np
import gpaw.mpi as mpi
from ase.units import Bohr
from .coulomb_core import CoulombCalculatorBase
from . import eri_utils as utils


class IjjiCoulombCalculator(CoulombCalculatorBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", "ijji")
        super().__init__(*args, **kwargs)

    def run(self):
        V = np.zeros((2 * self.Rx + 1, 2 * self.Ry + 1, 2 * self.Rz + 1, self.num_wann, self.num_wann))
        for w_i in self.pool_wf_indices:
            wf_loaded_i = utils.load_and_normalize_wf(self.npy_files[w_i], self.dV)
            wf_interpolated_i = self.map_wf_to_poisson(wf_loaded_i, self.grid, method=self.interp_method)
            for w_j in self.rank_wf_indices:
                wf_loaded_j = utils.load_and_normalize_wf(self.npy_files[w_j], self.dV)
                wf_interpolated_j = self.map_wf_to_poisson(wf_loaded_j, self.grid, method=self.interp_method)
                shift_list = []
                for shift in np.ndindex(2 * self.Rx - 1, 2 * self.Ry - 1, 2 * self.Rz - 1):
                    shift = np.array([self.Rx - 1, self.Ry - 1, self.Rz - 1]) - np.array(shift)
                    if np.array([(shift == s).all() for s in shift_list]).any():
                        continue
                    wf_shifted_j = utils.shift_WF(wf_interpolated_j, shift[0] * self.n_grid_uc[0], shift[1] * self.n_grid_uc[1], shift[2] * self.n_grid_uc[2])
                    coulomb_potential = self.GD.empty()
                    from gpaw.poisson import PoissonSolver as _PS
                    poisson_local = _PS(name="fast", nn=3)
                    poisson_local.set_grid_descriptor(self.GD)
                    poisson_local.solve(coulomb_potential, wf_interpolated_i * wf_shifted_j, charge=None, zero_initial_phi=False)
                    V[shift[0], shift[1], shift[2], w_i, w_j] = np.vdot(wf_interpolated_i * wf_shifted_j, coulomb_potential)
                    V[-shift[0], -shift[1], -shift[2], w_j, w_i] = V[shift[0], shift[1], shift[2], w_i, w_j]

        mpi.world.sum(V)
        V = V / Bohr
        epsilon_0 = 8.854187817e-12
        e = 1.602176634e-19
        conversion_factor = e / (4 * np.pi * epsilon_0) * 1e10
        V *= conversion_factor

        if not mpi.world.rank:
            output_cfg = self.config.get("output", {})
            save_dir = output_cfg.get("save_dir", "./")
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception:
                pass
            filename = output_cfg.get("matrix_filename", "V_matrix_ijji.npy")
            path = os.path.join(save_dir, filename)
            if output_cfg.get("use_unique_filenames", True):
                path = utils.uniquify(path)
            np.save(path, V)
        return V

