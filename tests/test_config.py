import pytest

from coulomb_matrix.config import CoulombConfig


def test_config_uses_defaults_and_resolves_relative_paths(tmp_path):
    config_file = tmp_path / "coulomb_config.toml"
    config_file.write_text(
        """
[paths]
xsf_dir = "xsf-files"
npy_dir = "npy-files"

[output]
save_dir = "outputs"

[interaction]
Rz = 2
"""
    )

    config, resolved_file = CoulombConfig.from_toml(config_file)

    assert resolved_file == config_file.resolve()
    assert config.paths.xsf_dir == (tmp_path / "xsf-files").resolve()
    assert config.paths.npy_dir == (tmp_path / "npy-files").resolve()
    assert config.output.save_dir == (tmp_path / "outputs").resolve()
    assert config.output.matrix_filename == "coulomb_matrix.npy"
    assert config.output.exchange_matrix_filename == "exchange_matrix.npy"
    assert config.interaction.Rx == 0
    assert config.interaction.Ry == 0
    assert config.interaction.Rz == 2
    assert config.mpi.number_poisson_pools == 1
    assert config.io.xsf_glob == "wannier90*.xsf"
    assert config.io.npy_glob == "wannier90*.npy"


def test_config_rejects_negative_interaction_range(tmp_path):
    config_file = tmp_path / "coulomb_config.toml"
    config_file.write_text(
        """
[interaction]
Rx = -1
"""
    )

    with pytest.raises(ValueError, match="interaction.Rx"):
        CoulombConfig.from_toml(config_file)
