import itertools

from coulomb_matrix.mpi_utils import compute_mpi_distribution


def collect_pool_indices(mpi_size, num_wann, number_poisson_pools):
    pools = {}
    for rank in range(mpi_size):
        dist = compute_mpi_distribution(mpi_size, rank, num_wann, number_poisson_pools)
        pool = dist['poisson_pool']
        pools.setdefault(pool, set()).update(dist['pool_wf_indices'])
    return pools


def test_pools_cover_all_indices():
    mpi_size = 8
    num_wann = 20
    number_poisson_pools = 3

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools)
    all_indices = set().union(*pools.values())
    assert all_indices == set(range(num_wann))


def test_no_overlap_between_pools():
    mpi_size = 7
    num_wann = 13
    number_poisson_pools = 4

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools)
    # ensure pairwise disjoint
    sets = list(pools.values())
    for a, b in itertools.combinations(sets, 2):
        assert a.isdisjoint(b)


def test_indices_within_bounds_and_consistent():
    for mpi_size, num_wann, pools in [
        (1, 1, 1),
        (4, 10, 2),
        (8, 100, 4),
        (16, 5, 3),
    ]:
        for rank in range(mpi_size):
            dist = compute_mpi_distribution(mpi_size, rank, num_wann, pools)
            # indices are within global range
            assert all(0 <= i < num_wann for i in dist['pool_wf_indices'])
            assert all(0 <= i < num_wann for i in dist['rank_wf_indices'])
            # no duplicates within assigned lists
            assert len(dist['pool_wf_indices']) == len(set(dist['pool_wf_indices']))
            assert len(dist['rank_wf_indices']) == len(set(dist['rank_wf_indices']))


def test_even_distribution_when_possible():
    mpi_size = 6
    num_wann = 12
    number_poisson_pools = 3

    pools = collect_pool_indices(mpi_size, num_wann, number_poisson_pools)
    sizes = [len(s) for s in pools.values()]
    # should be equal sized here
    assert set(sizes) == {4}
