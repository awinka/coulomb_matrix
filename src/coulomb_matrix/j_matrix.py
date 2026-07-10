"""Coulomb exchange calculator subclass and CLI entrypoint.

Provides `JCoulombCalculator` and a `main()` entrypoint.
"""

import numpy as np
from .coulomb_core import CoulombCalculatorBase
from .eri_utils import load_and_normalize_wf, shift_WF, convert_to_ev


class JCoulombCalculator(CoulombCalculatorBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", "ijji")
        super().__init__(*args, **kwargs)
        self.prev_w_i = None
        self.prev_w_j = None

    def run(self):
        # NOTE: This is not very memory efficient.
        V = np.zeros((2 * self.Rx + 1, 2 * self.Ry + 1, 2 * self.Rz + 1, self.num_wann, self.num_wann))
        for w_i, w_j in self.pool_pair_indices:
            # Check if w_i or w_j is the same as in the previous iteration to avoid redundant loading and interpolation
            if w_i != getattr(self, "prev_w_i", None):
                wf_i = load_and_normalize_wf(self.npy_files[w_i], self.dV)
                wf_i = self.map_wf_to_poisson(wf_i, self.grid, method=self.interp_method)
                self.prev_w_i = w_i
            if w_j != getattr(self, "prev_w_j", None):
                wf_j = load_and_normalize_wf(self.npy_files[w_j], self.dV)
                wf_j = self.map_wf_to_poisson(wf_j, self.grid, method=self.interp_method)
                self.prev_w_j = w_j
            shift_list = []
            for shift in np.ndindex(2 * self.Rx - 1, 2 * self.Ry - 1, 2 * self.Rz - 1):
                shift = np.array([self.Rx - 1, self.Ry - 1, self.Rz - 1]) - np.array(shift)
                if np.array([(shift == s).all() for s in shift_list]).any():
                    # Utilize symmetry to avoid redundant calculations
                    continue
                wf_shifted_j = shift_WF(wf_j, shift[0] * self.n_grid_uc[0], shift[1] * self.n_grid_uc[1], shift[2] * self.n_grid_uc[2])
                local_coulomb_potential = self.GD.zeros()
                self.poisson.solve(local_coulomb_potential, wf_i * wf_shifted_j.conj()) # , charge=None, zero_initial_phi=False)
                V[shift[0], shift[1], shift[2], w_i, w_j] = np.vdot(wf_i * wf_shifted_j.conj(), local_coulomb_potential)
                V[-shift[0], -shift[1], -shift[2], w_j, w_i] = V[shift[0], shift[1], shift[2], w_i, w_j]
                shift_list.append(-shift)

        self.comm.sum(V)
        convert_to_ev(V)
        return V
