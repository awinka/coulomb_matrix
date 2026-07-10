#!/usr/bin/env python3
"""Utility routines previously in ERI_utils_mod.py

Renamed for clarity and packaged under `coulomb_matrix.eri_utils`.
"""
import os

import gpaw.mpi as mpi
import numpy as np
from ase.units import Bohr


def shift_WF(x, nx, ny, nz):
    padding = (nx, ny, nz)
    x_shifted = x.copy()
    for i, p in enumerate(padding):
        padds = [[0, 0], [0, 0], [0, 0]]
        slice_idx = [slice(None)] * 3
        if np.sign(p) == 1:
            padds[i][0] = p
            slice_idx[i] = slice(-p)
            slice_idx = tuple(slice_idx)
            x_shifted = np.pad(x_shifted, padds, "constant")[slice_idx]
        elif np.sign(p) == -1:
            padds[i][1] = -p
            slice_idx[i] = slice(-p, None)
            slice_idx = tuple(slice_idx)
            x_shifted = np.pad(x_shifted, padds, "constant")[slice_idx]
        else:
            continue
    return x_shifted


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1

    return path


def load_and_normalize_wf(npy_file, dV):
    with open(npy_file, "rb") as npy:
        wf_loaded = np.load(npy)
        wf_loaded = wf_loaded / np.sqrt(np.vdot(wf_loaded.conj(), wf_loaded) * dV)
    return wf_loaded


def convert_to_ev(V):
    """Convert the Coulomb matrix V from atomic units to electron volts (eV)."""
    V /= Bohr
    epsilon_0 = 8.854187817e-12
    e = 1.602176634e-19
    conversion_factor = e / (4 * np.pi * epsilon_0) * 1e10
    V *= conversion_factor
