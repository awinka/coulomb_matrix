"""Shared Coulomb matrix computation core for `ijij` and `ijji` modes.

This module provides `load_config` and `CoulombCalculatorBase` which
encapsulates the common MPI/grid/IO setup. Mode-specific behavior is
implemented as two helper methods inside the class.
"""
from pathlib import Path
import tomllib
import glob
import re
import gpaw.mpi as mpi
from gpaw.poisson import PoissonSolver
from .xsf import Xsf2Np
from .mpi_utils import compute_mpi_distribution
from .grid_utils import PoissonGrid


def load_config(config_path):
    config_file = None
    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
    else:
        default_path = Path(__file__).with_name("coulomb_config.toml")
        if default_path.exists():
            config_file = default_path.resolve()

    raw_config = {}
    if config_file is not None:
        with open(config_file, "rb") as f:
            raw_config = tomllib.load(f)

    config = raw_config

    # Lightweight validation / normalization
    interaction = config.get("interaction", {})
    grid = config.get("grid", {})
    mpi_cfg = config.get("mpi", {})
    paths = config.get("paths", {})
    io_cfg = config.get("io", {})
    output = config.get("output", {})
    runtime = config.get("runtime", {})

    for key in ("Rx", "Ry", "Rz"):
        if key in interaction:
            interaction[key] = int(interaction[key])
            if interaction[key] < 1:
                raise ValueError(f"Config field 'interaction.{key}' must be >= 1.")
        else:
            interaction[key] = 1

    # remove resolution handling; grids will be derived directly from XSF data

    mpi_cfg["number_poisson_pools"] = int(mpi_cfg.get("number_poisson_pools", 1))
    if mpi_cfg["number_poisson_pools"] < 1:
        raise ValueError("Config field 'mpi.number_poisson_pools' must be >= 1.")

    runtime["perform_checks"] = bool(runtime.get("perform_checks", False))
    runtime["omp_num_threads"] = int(runtime.get("omp_num_threads", 1))
    output["use_unique_filenames"] = bool(output.get("use_unique_filenames", True))

    # Ensure paths/IO strings exist (may be empty if user relies on defaults)
    for section in (paths, output, io_cfg):
        for k, v in list(section.items()):
            if isinstance(v, str):
                section[k] = v.strip()

    config.update(
        {
            "interaction": interaction,
            "grid": grid,
            "mpi": mpi_cfg,
            "paths": paths,
            "io": io_cfg,
            "output": output,
            "runtime": runtime,
        }
    )

    return config, config_file


class CoulombCalculatorBase:
    def __init__(
        self,
        config_path=None,
        mode="ijij",
        xsf_dir: str | None = None,
        npy_dir: str | None = None,
        seedname: str | None = None,
        pattern_xsf: str | None = None,
        pattern_npy: str | None = None,
    ):
        if mode not in ("ijij", "ijji"):
            raise ValueError("mode must be 'ijij' or 'ijji'")
        self.mode = mode
        self.config, self.config_file = load_config(config_path)
        self._override_xsf_dir = xsf_dir
        self._override_npy_dir = npy_dir
        self._override_seedname = seedname
        self._override_pattern_xsf = pattern_xsf
        self._override_pattern_npy = pattern_npy
        # Shared initializations used by subclasses
        self.comm = mpi.world
        self.rank = self.comm.rank
        self.mpi_size = self.comm.size

        config = self.config
        self.paths_cfg = config.get("paths", {})
        self.interaction_cfg = config.get("interaction", {})
        self.grid_cfg = config.get("grid", {})
        self.mpi_cfg = config.get("mpi", {})
        self.io_cfg = config.get("io", {})

        # Derived file paths and patterns
        xsf_path = (self._override_xsf_dir or self.paths_cfg.get("xsf_dir", "")) + (
            self._override_seedname or self.paths_cfg.get("seedname", "")
        )
        npy_path = (self._override_npy_dir or self.paths_cfg.get("npy_dir", "")) + (
            self.paths_cfg.get("npy_prefix", "") or ""
        )
        pattern_xsf = self._override_pattern_xsf or self.io_cfg.get("xsf_glob", "*.xsf")
        pattern_npy = self._override_pattern_npy or self.io_cfg.get("npy_glob", "*.npy")

        self.npy_files = glob.glob(npy_path + pattern_npy)
        self.npy_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        self.xsf_files = glob.glob(xsf_path + pattern_xsf)
        if self.npy_files == [] or self.xsf_files == []:
            raise ValueError("No npy or xsf files found.")

        self.Rx = int(self.interaction_cfg.get("Rx", 1))
        self.Ry = int(self.interaction_cfg.get("Ry", 1))
        self.Rz = int(self.interaction_cfg.get("Rz", 1))

        self.num_wann = len(self.npy_files)

        dist = compute_mpi_distribution(
            mpi_size=self.mpi_size,
            rank=self.rank,
            num_wann=self.num_wann,
            n_poisson_pools=int(self.mpi_cfg.get("number_poisson_pools", 1)),
            mode=self.mode,
        )

        if self.mode == "ijij":
            self.pool_wf_indices = dist["pool_wf_indices"]
            self.rank_wf_indices = dist["rank_wf_indices"]
        elif self.mode == "ijji":
            self.pool_pair_indices = dist["pool_pair_indices"]
        self.comm_poisson = self.comm.new_communicator(dist["comm_poisson_ranks"])

        # Build Poisson grid
        # Read one of the XSF files to get the grid descriptor for the Poisson solver
        if self.comm.rank == 0:
            with open(self.xsf_files[0], "r") as xsf:
                wf_file = Xsf2Np(xsf, skip_wf=True)
            lattice_vectors = wf_file.get_lattice_vectors()
            supercell_vectors = wf_file.get_supercell()
            real_space_grid = np.array(wf_file.get_real_space_grid())
            self.comm.broadcast(lattice_vectors, 0)
            self.comm.broadcast(supercell_vectors, 0)
            self.comm.broadcast(real_space_grid, 0)
        else:
            lattice_vectors = self.comm.broadcast(None, 0)
            supercell_vectors = self.comm.broadcast(None, 0)
            real_space_grid = self.comm.broadcast(None, 0)

        # Very slow initialization, can I make it faster?
        self.pg = PoissonGrid(lattice_vectors, supercell_vectors, real_space_grid, self.comm_poisson, self.interaction_cfg)

        # commonly used objects
        self.dV = self.pg.dV
        self.n_grid_uc = self.pg.n_grid_uc
        self.GD = self.pg.GD
        self.grid = self.pg.grid
        self.grid_full = self.pg.grid_full
        self.map_wf_to_poisson = self.pg.map_wf_to_poisson
        # This should not matter. WF is just put on a larger grid with the same spacing, 
        # so nearest neighbor interpolation should be sufficient. I believe it is faster than linear interpolation, 
        # but this should be tested.
        self.interp_method = "nearest" # or "linear"

        # Build Poisson solver
        # Very slow initialization, can I make it faster? 
        self.poisson = PoissonSolver(name="fast", nn=3)
        self.poisson.set_grid_descriptor(self.GD)

    def run(self):
        raise NotImplementedError("Subclasses must implement run().")
