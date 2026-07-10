"""Coulomb matrix calculator subclass and CLI entrypoint.

Provides `VCoulombCalculator` and a `main()` entrypoint.
"""

import numpy as np

from .coulomb_core import CoulombCalculatorBase
from .eri_utils import convert_to_ev, load_and_normalize_wf, shift_WF


class VCoulombCalculator(CoulombCalculatorBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", "ijij")
        super().__init__(*args, **kwargs)

    def run(self):
        # Prepare output matrix container
        # NOTE: This is not very memory efficient.
        V = np.zeros(
            (
                2 * self.Rx + 1,
                2 * self.Ry + 1,
                2 * self.Rz + 1,
                self.num_wann,
                self.num_wann,
            )
        )

        for w_i in self.pool_wf_indices:
            wf_i = load_and_normalize_wf(self.npy_files[w_i], self.dV)
            # interpolate WF to Poisson grid
            wf_i = self.map_wf_to_poisson(wf_i, self.grid)

            coulomb_potential = self.solve_poisson(wf_i.conj() * wf_i)

            for w_j in self.rank_wf_indices:
                wf_j = load_and_normalize_wf(self.npy_files[w_j], self.dV)
                wf_j = self.map_wf_to_poisson(
                    wf_j, self.grid_full, method=self.interp_method
                )
                # TODO: Could exploit symmetry here to avoid redundant calculations.
                for shift in np.ndindex(
                    2 * self.Rx + 1, 2 * self.Ry + 1, 2 * self.Rz + 1
                ):
                    shift = np.array([self.Rx, self.Ry, self.Rz]) - np.array(shift)
                    wf_shifted_j = shift_WF(
                        wf_j,
                        shift[0] * self.n_grid_uc[0],
                        shift[1] * self.n_grid_uc[1],
                        shift[2] * self.n_grid_uc[2],
                    )
                    V[shift[0], shift[1], shift[2], w_i, w_j] = (
                        np.vdot(wf_shifted_j.conj() * wf_shifted_j, coulomb_potential)
                        * self.GD.dv
                    )

        self.comm.sum(V)
        convert_to_ev(V)
        return V
