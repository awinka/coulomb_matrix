"""Typed configuration objects for Coulomb matrix calculations.

The configuration is split into small dataclasses so the available input
variables, their defaults, and their meaning live in one place.

Sections:
- paths: locations of the input Wannier files and output directory
- output: filename handling for saved Coulomb matrices
- interaction: interaction range in unit-cell steps
- mpi: MPI distribution settings
- io: glob patterns used to discover Wannier files
"""

from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
from typing import Any, Mapping

import tomllib


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    coerced_value = int(value)
    if coerced_value < 0:
        raise ValueError(f"Config field '{field_name}' must be >= 0.")
    return coerced_value


def _resolve_path(value: Any, config_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


@dataclass(slots=True)
class PathsConfig:
    """Input file locations.

    Attributes
    ----------
    xsf_dir:
        Directory containing the Wannier `.xsf` files.
    npy_dir:
        Directory containing the corresponding Wannier `.npy` files.
    """

    xsf_dir: Path = field(
        metadata={"help": "Directory containing the Wannier XSF files."}
    )
    npy_dir: Path = field(
        metadata={"help": "Directory containing the Wannier NPY files."}
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, config_dir: Path
    ) -> "PathsConfig":
        data = data or {}
        return cls(
            xsf_dir=_resolve_path(data.get("xsf_dir", config_dir), config_dir),
            npy_dir=_resolve_path(data.get("npy_dir", config_dir), config_dir),
        )


@dataclass(slots=True)
class OutputConfig:
    """Output file settings.

    Attributes
    ----------
    save_dir:
        Directory where the resulting matrix is written.
    matrix_filename:
        Base filename for the saved matrix.
    exchange_matrix_filename:
        Base filename used by the exchange-matrix CLI.
    use_unique_filenames:
        When true, existing output paths are uniquified before saving.
    """

    save_dir: Path = field(metadata={"help": "Directory where the matrix is saved."})
    matrix_filename: str = field(
        default="coulomb_matrix.npy",
        metadata={"help": "Filename used for the Coulomb matrix output."},
    )
    exchange_matrix_filename: str = field(
        default="exchange_matrix.npy",
        metadata={"help": "Filename used for the exchange Coulomb matrix output."},
    )
    use_unique_filenames: bool = field(
        default=False,
        metadata={"help": "Append a suffix if the output file already exists."},
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, config_dir: Path
    ) -> "OutputConfig":
        data = data or {}
        return cls(
            save_dir=_resolve_path(data.get("save_dir", config_dir), config_dir),
            matrix_filename=str(data.get("matrix_filename", "coulomb_matrix.npy")),
            exchange_matrix_filename=str(
                data.get("exchange_matrix_filename", "exchange_matrix.npy")
            ),
            use_unique_filenames=bool(data.get("use_unique_filenames", False)),
        )


@dataclass(slots=True)
class InteractionConfig:
    """Interaction range measured in unit-cell steps.

    Rx, Ry, and Rz determine how many neighboring unit cells are included in
    each lattice direction. A value of zero keeps the calculation local in that
    direction.
    """

    Rx: int = field(
        default=0, metadata={"help": "Interaction range along x in unit-cell steps."}
    )
    Ry: int = field(
        default=0, metadata={"help": "Interaction range along y in unit-cell steps."}
    )
    Rz: int = field(
        default=0, metadata={"help": "Interaction range along z in unit-cell steps."}
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, config_dir: Path | None = None
    ) -> "InteractionConfig":
        data = data or {}
        return cls(
            Rx=_coerce_non_negative_int(data.get("Rx", 0), "interaction.Rx"),
            Ry=_coerce_non_negative_int(data.get("Ry", 0), "interaction.Ry"),
            Rz=_coerce_non_negative_int(data.get("Rz", 0), "interaction.Rz"),
        )


@dataclass(slots=True)
class MpiConfig:
    """MPI scheduling parameters.

    number_poisson_pools controls how many Poisson-solver pools the work is
    split into.
    """

    number_poisson_pools: int = field(
        default=1,
        metadata={"help": "Number of Poisson pools used for MPI distribution."},
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, config_dir: Path | None = None
    ) -> "MpiConfig":
        data = data or {}
        number_poisson_pools = int(data.get("number_poisson_pools", 1))
        if number_poisson_pools < 1:
            raise ValueError("Config field 'mpi.number_poisson_pools' must be >= 1.")
        return cls(number_poisson_pools=number_poisson_pools)


@dataclass(slots=True)
class IoConfig:
    """File-discovery patterns for Wannier inputs."""

    xsf_glob: str = field(
        default="wannier90*.xsf",
        metadata={"help": "Glob pattern used to collect XSF Wannier files."},
    )
    npy_glob: str = field(
        default="wannier90*.npy",
        metadata={"help": "Glob pattern used to collect NPY Wannier files."},
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, config_dir: Path | None = None
    ) -> "IoConfig":
        data = data or {}
        return cls(
            xsf_glob=str(data.get("xsf_glob", "wannier90*.xsf")),
            npy_glob=str(data.get("npy_glob", "wannier90*.npy")),
        )


@dataclass(slots=True)
class CoulombConfig:
    """Complete Coulomb matrix configuration."""

    paths: PathsConfig = field(metadata={"help": "Input file locations."})
    output: OutputConfig = field(metadata={"help": "Output file settings."})
    interaction: InteractionConfig = field(
        metadata={"help": "Interaction range settings."}
    )
    mpi: MpiConfig = field(metadata={"help": "MPI scheduling settings."})
    io: IoConfig = field(metadata={"help": "File-discovery patterns."})

    @classmethod
    def get_help_text(cls, target_cls: Any = None, prefix: str = "") -> str:
        """Recursively generates a help menu from the configuration schema."""
        if target_cls is None:
            target_cls = cls

        lines = []
        if not prefix:
            lines.append("Available TOML Configuration Options:")
            lines.append("=" * 37 + "\n")

        for f in fields(target_cls):
            # Formulate the TOML path (e.g., paths.xsf_dir)
            current_path = f"{prefix}.{f.name}" if prefix else f.name

            if is_dataclass(f.type):
                section_help = f.metadata.get("help", "Configuration section.")
                lines.append(f"[{current_path}]")
                lines.append(f"    {section_help}\n")
                # Recurse down into the nested dataclass
                lines.append(cls.get_help_text(f.type, prefix=current_path))
            else:
                help_text = f.metadata.get("help", "No description provided.")
                if f.default == MISSING:
                    default_str = " (Required)"
                else:
                    # Formats strings with quotes, booleans nicely, etc.
                    default_str = f" (Default: {repr(f.default)})"

                # Get the type name safely whether it's a class or a string
                if isinstance(f.type, str):
                    type_name = f.type
                elif hasattr(f.type, "__name__"):
                    type_name = f.type.__name__
                else:
                    type_name = str(f.type) # Fallback for complex types like List[str] or Optional[int]

                lines.append(f"  {f.name} [{type_name}]{default_str}")
                lines.append(f"        {help_text}\n")

        return "\n".join(lines)
    
    @classmethod
    def from_dict(
        cls, raw_config: Mapping[str, Any] | None, config_dir: Path
    ) -> "CoulombConfig":
        raw_config = raw_config or {}
        return cls(
            paths=PathsConfig.from_mapping(raw_config.get("paths"), config_dir),
            output=OutputConfig.from_mapping(raw_config.get("output"), config_dir),
            interaction=InteractionConfig.from_mapping(raw_config.get("interaction")),
            mpi=MpiConfig.from_mapping(raw_config.get("mpi")),
            io=IoConfig.from_mapping(raw_config.get("io")),
        )

    @classmethod
    def from_toml(
        cls, config_path: str | Path | None = None
    ) -> tuple["CoulombConfig", Path | None]:
        config_file: Path | None = None
        if config_path:
            config_file = Path(config_path).expanduser().resolve()
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
        else:
            default_path = Path.cwd() / "coulomb_config.toml"
            if default_path.exists():
                config_file = default_path.resolve()

        config_dir = config_file.parent if config_file else Path.cwd()
        raw_config: Mapping[str, Any] = {}
        if config_file is not None:
            with open(config_file, "rb") as file_handle:
                raw_config = tomllib.load(file_handle)

        return cls.from_dict(raw_config, config_dir), config_file
