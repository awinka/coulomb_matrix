"""Coulomb matrix calculator subclass and CLI entrypoint.

Provides `VCoulombCalculator` and a `main()` entrypoint.
"""

import numpy as np
import gpaw.mpi as mpi
from .coulomb_core import CoulombCalculatorBase
from .eri_utils import load_and_normalize_wf, shift_WF, convert_to_ev


class VCoulombCalculator(CoulombCalculatorBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", "ijij")
        super().__init__(*args, **kwargs)

    def run(self):
        # Prepare output matrix container
        # NOTE: This is not very memory efficient.
        V = np.zeros((2 * self.Rx + 1, 2 * self.Ry + 1, 2 * self.Rz + 1, self.num_wann, self.num_wann))

        local_coulomb_potential = self.GD.zeros()
        coulomb_potential = self.GD.zeros(global_array=True)
        for w_i in self.pool_wf_indices:
            wf_i = load_and_normalize_wf(self.npy_files[w_i], self.dV)
            # interpolate WF to Poisson grid
            wf_i = self.map_wf_to_poisson(wf_i, self.grid, method=self.interp_method)

            local_coulomb_potential[:] = 0.0
            self.poisson.solve(local_coulomb_potential, wf_i.conj() * wf_i)

            # Check if this is really necessary or if gpaw provides functionality
            coulomb_potential[:] = 0.0
            coulomb_potential[self.GD.beg_c[0]:self.GD.end_c[0], self.GD.beg_c[1]:self.GD.end_c[1], self.GD.beg_c[2]:self.GD.end_c[2]] = local_coulomb_potential
            self.GD.comm.sum(coulomb_potential)

            for w_j in self.rank_wf_indices:
                wf_j = load_and_normalize_wf(self.npy_files[w_j], self.dV)
                wf_j = self.map_wf_to_poisson(wf_j, self.grid_full, method=self.interp_method)
                shift_list = []
                for shift in np.ndindex(2 * self.Rx + 1, 2 * self.Ry + 1, 2 * self.Rz + 1):
                    shift = np.array([self.Rx, self.Ry, self.Rz]) - np.array(shift)
                    if np.array([(shift == s).all() for s in shift_list]).any():
                        # Utilize symmetry to avoid redundant calculations
                        continue
                    wf_shifted_j = shift_WF(wf_j, shift[0] * self.n_grid_uc[0], shift[1] * self.n_grid_uc[1], shift[2] * self.n_grid_uc[2])
                    V[shift[0], shift[1], shift[2], w_i, w_j] = np.vdot(wf_shifted_j.conj() * wf_shifted_j, coulomb_potential) * self.GD.dv
                    V[-shift[0], -shift[1], -shift[2], w_j, w_i] = V[shift[0], shift[1], shift[2], w_i, w_j]
                    shift_list.append(-shift)

        mpi.world.sum(V)
        return convert_to_ev(V)

