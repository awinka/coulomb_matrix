# justfile for common developer tasks
# Usage:
#   just fmt          # format code
#   just lint         # lint code
#   just tests        # run pytest (single-process)
#   just tests-mpi    # run pytest under MPI

set shell := ["bash", "-cu"]

# default number of MPI ranks for MPI test target
N := 4

default: tests

fmt:
	# Apply import sorting + formatting and apply ruff's formatter
	isort --profile black .
	black .
	ruff format .

lint:
	# Static linter (ruff) and optional flake8 for extra checks
	ruff check .
	flake8 .

fmt-check:
	# Check formatting without modifying files
	isort --check-only --profile black .
	black --check .
	ruff check .

tests:
	# Run pytest in the current Python environment
	python -m pytest -q

tests-mpi:
	# Run tests using MPI. Adjust N at top or pass like `just tests-mpi N=2`.
	mpiexec -n {{N}} python -m pytest -q

tests-all:
	# Run both single-process and MPI tests (MPI uses N ranks).
	python -m pytest -q
	mpiexec -n {{N}} python -m pytest -q
