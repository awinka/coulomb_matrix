# Instructions

There are 3 available commands; `xsf-to-npy`, `coulomb-matrix`, and `exchange-matrix`.

## `xsf-to-npy`

Wannier90 outputs the Wannier functions on a real-space grid as xsf-files. Currently in the
code, the Wannier function files are read many times. To speed up the parsing of the Wannier functions,
npy-files are created from the xsf files. This is done with the command `xsf-to-npy`. Type `xsf-to-npy --help`
see available flags.

## `coulomb-matrix`

This command calculates the Coulomb matrix from the Wannier functions. As inputs you need the Wannier `.npy` files and `.xsf` files (one is sufficient). There is also a config file `coulomb_config.toml`. The easiest way to see available options is to look in the `src/config.py`  file. The documentation might improve in the future. 

## `exchange-matrix`

Command for calculating the exchange matrix. It uses the same inputs and config file as `coulomb-matrix`.

# Running the example

In this folder there are currently three `xsf` Wannier files in the `input-files` folder. To run the example, first convert the xsf files to npy files with the command:

```bash
cd input-files
xsf-to-npy
cd ..
```

Then run the command to calculate the Coulomb matrix:

```bash
coulomb-matrix --config coulomb_config.toml
```

The code also supports MPI, so you can run it with multiple processes. For example, to run with 4 processes:

```bash
mpirun -np 4 coulomb-matrix --config coulomb_config.toml
```

The output will be a file called `coulomb_matrix.npy` in the current directory. The exchange matrix can be calculated with the command:

```bash
coulomb-exchange --config coulomb_config.toml
```

or

```bash
mpirun -np 4 coulomb-exchange --config coulomb_config.toml
```

The exchange calculation is typically much slower, since you have to solve the Poisson equation for all pairs of Wannier functions.