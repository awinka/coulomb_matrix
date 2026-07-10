import math

import numpy as np


def compute_mpi_distribution(
    mpi_size: int,
    rank: int,
    num_wann: int,
    n_poisson_pools: int | None = None,
    mode: str = "ijij",
):
    """Compute distribution of Wannier-function indices across MPI ranks.

    Parameters
    - mpi_size: total number of MPI ranks
    - rank: current MPI rank
    - num_wann: total number of Wannier functions (global indices 0..num_wann-1)
    - n_poisson_pools: desired number of independent Poisson solver pools
    - mode: "ijij" or "ijji", determines how the distribution is done. In "ijij" mode, each pool gets a subset of Wannier functions, and each rank within the pool gets a further subset. In "ijji" mode, each pool gets a subset of ij pairs. This means that currently, the ijji mode scales better with the number of MPI ranks.

    Returns a dict with:
    - "ranks_per_pool": number of MPI ranks assigned to each Poisson pool
    - "poisson_pool": pool index this rank belongs to
    - "comm_poisson_ranks": list of ranks in the same pool
    - "new_rank": rank index within the pool
    Mode-specific:
    - "ijij":
        - "num_wann_per_pool": number of Wannier functions assigned per pool
        - "num_wann_per_rank": number of Wannier functions assigned per rank (within the pool)
        - "pool_wf_indices": global WF indices assigned to this pool
        - "rank_wf_indices": global WF indices assigned to this rank (subset of pool_wf_indices)
    - "ijji":
        - "num_pairs_per_pool": number of ij pairs assigned per pool
        - "num_pairs_per_rank": number of ij pairs assigned per rank (within the pool)
        - "pool_pair_indices": global ij pair indices assigned to this pool

    The allocation tries to distribute "num_wann" evenly across pools and then
    evenly across ranks inside each pool.
    """
    if n_poisson_pools is None or n_poisson_pools < 1:
        n_poisson_pools = 1

    # The number of pools cannot exceed the number of MPI ranks or the number of Wannier functions (or pairs)
    # Instead of raising an error, we will just reduce the number of pools to the maximum possible.
    # We should however warn the user that they requested more pools than possible.
    if n_poisson_pools > mpi_size:
        print(
            f"Warning: Requested {n_poisson_pools} Poisson pools, but only {mpi_size} MPI ranks are available. Reducing to {mpi_size}."
        )
    if mode == "ijij" and n_poisson_pools > num_wann:
        print(
            f"Warning: Requested {n_poisson_pools} Poisson pools, but only {num_wann} Wannier functions are available. Reducing to {num_wann}."
        )
    if mode == "ijji" and n_poisson_pools > num_wann * num_wann:
        print(
            f"Warning: Requested {n_poisson_pools} Poisson pools, but only {num_wann * num_wann} ij pairs are available. Reducing to {num_wann * num_wann}."
        )

    n_poisson_pools = min(
        n_poisson_pools, mpi_size, num_wann if mode == "ijij" else num_wann * num_wann
    )

    ranks_per_pool = math.ceil(mpi_size / n_poisson_pools)
    poisson_pool = rank // ranks_per_pool

    # ranks that belong to this poisson pool
    comm_poisson_ranks = [
        q for q in range(mpi_size) if q // ranks_per_pool == poisson_pool
    ]
    try:
        new_rank = comm_poisson_ranks.index(rank)
    except ValueError:
        new_rank = 0

    return_dict = {
        "ranks_per_pool": ranks_per_pool,
        "poisson_pool": poisson_pool,
        "comm_poisson_ranks": comm_poisson_ranks,
        "new_rank": new_rank,
    }

    if mode == "ijij":
        num_wann_per_pool = (
            math.ceil(num_wann / n_poisson_pools) if n_poisson_pools else num_wann
        )
        pool_start = poisson_pool * num_wann_per_pool
        pool_end = min((poisson_pool + 1) * num_wann_per_pool, num_wann)
        pool_wf_indices = list(range(pool_start, pool_end))

        num_wann_per_rank = math.ceil(num_wann / max(1, len(comm_poisson_ranks)))
        rank_start = new_rank * num_wann_per_rank
        rank_end = min((new_rank + 1) * num_wann_per_rank, num_wann)
        rank_wf_indices = [i for i in range(rank_start, rank_end)]
        return_dict.update(
            {
                "num_wann_per_pool": num_wann_per_pool,
                "num_wann_per_rank": num_wann_per_rank,
                "pool_wf_indices": pool_wf_indices,
                "rank_wf_indices": rank_wf_indices,
            }
        )

    elif mode == "ijji":
        num_pairs_per_pool = (
            math.ceil(num_wann * num_wann / n_poisson_pools)
            if n_poisson_pools
            else num_wann * num_wann
        )
        pool_start = poisson_pool * num_pairs_per_pool
        pool_end = min((poisson_pool + 1) * num_pairs_per_pool, num_wann * num_wann)
        wf_indices = np.ndindex(num_wann, num_wann)
        pool_pair_indices = [
            pair for i, pair in enumerate(wf_indices) if pool_start <= i < pool_end
        ]
        return_dict.update(
            {
                "num_pairs_per_pool": num_pairs_per_pool,
                "pool_pair_indices": pool_pair_indices,
            }
        )

    return return_dict
