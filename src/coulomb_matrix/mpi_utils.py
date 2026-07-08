import math


def compute_mpi_distribution(mpi_size: int, rank: int, num_wann: int, n_poisson_pools: int | None = None):
    """Compute distribution of Wannier-function indices across MPI ranks.

    Parameters
    - mpi_size: total number of MPI ranks
    - rank: current MPI rank
    - num_wann: total number of Wannier functions (global indices 0..num_wann-1)
    - n_poisson_pools: desired number of independent Poisson solver pools

    Returns a dict with:
    - "ranks_per_pool": number of MPI ranks assigned to each Poisson pool
    - "poisson_pool": pool index this rank belongs to
    - "comm_poisson_ranks": list of ranks in the same pool
    - "new_rank": rank index within the pool
    - "num_wann_per_pool": number of Wannier functions assigned per pool
    - "num_wann_per_rank": number of Wannier functions assigned per rank (within the pool)
    - "pool_wf_indices": global WF indices assigned to this pool
    - "rank_wf_indices": global WF indices assigned to this rank (subset of pool_wf_indices)

    The allocation tries to distribute "num_wann" evenly across pools and then
    evenly across ranks inside each pool.
    """
    if n_poisson_pools is None or n_poisson_pools < 1:
        n_poisson_pools = 1

    n_poisson_pools = min(n_poisson_pools, mpi_size)

    ranks_per_pool = math.ceil(mpi_size / n_poisson_pools)
    poisson_pool = rank // ranks_per_pool

    # ranks that belong to this poisson pool
    comm_poisson_ranks = [q for q in range(mpi_size) if q // ranks_per_pool == poisson_pool]
    try:
        new_rank = comm_poisson_ranks.index(rank)
    except ValueError:
        new_rank = 0

    num_wann_per_pool = math.ceil(num_wann / n_poisson_pools) if n_poisson_pools else num_wann
    pool_start = poisson_pool * num_wann_per_pool
    pool_end = min((poisson_pool + 1) * num_wann_per_pool, num_wann)
    pool_wf_indices = list(range(pool_start, pool_end))

    num_wann_per_rank = math.ceil(len(pool_wf_indices) / max(1, len(comm_poisson_ranks)))
    rank_start = new_rank * num_wann_per_rank
    rank_end = min((new_rank + 1) * num_wann_per_rank, len(pool_wf_indices))
    rank_wf_indices = [pool_wf_indices[i] for i in range(rank_start, rank_end)]

    return {
        "ranks_per_pool": ranks_per_pool,
        "poisson_pool": poisson_pool,
        "comm_poisson_ranks": comm_poisson_ranks,
        "new_rank": new_rank,
        "num_wann_per_pool": num_wann_per_pool,
        "num_wann_per_rank": num_wann_per_rank,
        "pool_wf_indices": pool_wf_indices,
        "rank_wf_indices": rank_wf_indices,
    }
