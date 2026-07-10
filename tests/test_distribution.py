"""
Some tests are probably superfluous (thanks though copilot), so in the future this file can be cleaned up
"""
import itertools

import pytest

from coulomb_matrix.mpi_utils import compute_mpi_distribution


def collect_pool_indices(mpi_size, num_wann, number_poisson_pools, mode="ijij"):
    pools = {}
    for rank in range(mpi_size):
        dist = compute_mpi_distribution(
            mpi_size, rank, num_wann, number_poisson_pools, mode=mode
        )
        pool = dist["poisson_pool"]
        if mode == "ijij":
            pools.setdefault(pool, set()).update(dist["pool_wf_indices"])
        else:  # "ijji"
            pools.setdefault(pool, set()).update(dist["pool_pair_indices"])
    return pools


def collect_rank_indices(mpi_size, num_wann, number_poisson_pools):
    # This is specific for the "ijij" mode, as "ijji" does not have rank_wf_indices
    ranks = {}
    poisson_pools = {}
    for rank in range(mpi_size):
        dist = compute_mpi_distribution(
            mpi_size, rank, num_wann, number_poisson_pools, mode="ijij"
        )
        poisson_pools[rank] = dist["poisson_pool"]
        ranks[rank] = dist["rank_wf_indices"]
    return ranks, poisson_pools


@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_pools_cover_all_indices(mode):
    mpi_size = 8
    num_wann = 20
    number_poisson_pools = 3

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools, mode=mode)
    all_indices = set().union(*pools.values())
    reference_indices = (
        set(range(num_wann))
        if mode == "ijij"
        else set((i, j) for i in range(num_wann) for j in range(num_wann))
    )
    assert all_indices == reference_indices


def test_ranks_cover_all_indices():
    mpi_size = 6
    num_wann = 12
    number_poisson_pools = 3

    ranks, poisson_pools = collect_rank_indices(
        mpi_size, num_wann, number_poisson_pools
    )
    for pool in range(number_poisson_pools):
        # Extract ranks that belong to this pool
        pool_ranks = [rank for rank, p in poisson_pools.items() if p == pool]
        # Collect all indices assigned to these ranks
        pool_indices = set().union(*(ranks[rank] for rank in pool_ranks))
        # Ensure that the rank indices within each pool span the range of all Wannier functions
        reference_indices = set(range(num_wann))
        assert pool_indices == reference_indices


@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_no_overlap_between_pools(mode):
    mpi_size = 7
    num_wann = 13
    number_poisson_pools = 4

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools, mode=mode)
    # ensure pairwise disjoint
    sets = list(pools.values())
    for a, b in itertools.combinations(sets, 2):
        assert a.isdisjoint(b)


def test_no_overlap_between_ranks():
    mpi_size = 5
    num_wann = 10
    number_poisson_pools = 2

    ranks, poisson_pools = collect_rank_indices(
        mpi_size, num_wann, number_poisson_pools
    )
    for pool in range(number_poisson_pools):
        # Extract ranks that belong to this pool
        pool_ranks = [rank for rank, p in poisson_pools.items() if p == pool]
        # Collect all indices assigned to these ranks
        rank_indices = [ranks[rank] for rank in pool_ranks]
        # ensure pairwise disjoint
        for a, b in itertools.combinations(rank_indices, 2):
            assert set(a).isdisjoint(set(b))


@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_indices_within_bounds_and_consistent(mode):
    for mpi_size, num_wann, pools in [
        (1, 1, 1),
        (4, 10, 2),
        (8, 100, 4),
        (16, 5, 3),
    ]:
        for rank in range(mpi_size):
            dist = compute_mpi_distribution(mpi_size, rank, num_wann, pools, mode=mode)
            if mode == "ijij":
                # indices are within global range
                assert all(0 <= i < num_wann for i in dist["pool_wf_indices"])
                assert all(0 <= i < num_wann for i in dist["rank_wf_indices"])
                # no duplicates within assigned lists
                assert len(dist["pool_wf_indices"]) == len(set(dist["pool_wf_indices"]))
                assert len(dist["rank_wf_indices"]) == len(set(dist["rank_wf_indices"]))
            else:  # "ijji"
                # indices are within global range
                assert all(
                    0 <= i < num_wann and 0 <= j < num_wann
                    for i, j in dist["pool_pair_indices"]
                )
                # no duplicates within assigned lists
                assert len(dist["pool_pair_indices"]) == len(
                    set(dist["pool_pair_indices"])
                )


@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_when_too_many_pools(mode):
    mpi_size = 10
    num_wann = 3
    number_poisson_pools = 10

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools, mode=mode)
    # The number of pools should be reduced to the maximum possible, which is num_wann for "ijij" and num_wann*num_wann for "ijji"
    expected_max_pools = num_wann if mode == "ijij" else num_wann * num_wann
    assert len(pools) <= expected_max_pools


@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_even_distribution_when_possible(mode):
    mpi_size = 6
    num_wann = 12
    number_poisson_pools = 3

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools, mode=mode)
    sizes = [len(s) for s in pools.values()]
    # should be equal sized here
    if mode == "ijij":
        assert set(sizes) == {4}
    else:  # "ijji"
        # For ijji, the number of pairs is num_wann*num_wann, which is 144. With 3 pools, each pool should ideally have 48 pairs.
        assert set(sizes) == {48}
