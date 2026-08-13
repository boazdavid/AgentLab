"""Deterministic train/test seed split for WorkArena-L1 + benchmark builder.

Interfaces consumed by later tasks (keep names/shapes stable):
- ``l1_task_ids() -> list[str]`` — the WorkArena-L1 atomic task ids.
- ``split_pairs(n_train=10, n_test=10, seed=42)`` — disjoint (task_id, seed) pairs.
- ``build_benchmark(pairs, max_steps=15) -> Benchmark`` — a workarena_l1 benchmark
  whose ``env_args_list`` is exactly ``pairs``.
"""

from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
from browsergym.experiments.loop import EnvArgs


def l1_task_ids() -> list[str]:
    """Return the sorted, distinct WorkArena-L1 atomic task ids.

    Enumerates task classes locally (no live ServiceNow instance required).
    """
    from browsergym.workarena import get_all_tasks_agents

    tuples = get_all_tasks_agents(
        filter="l1", meta_seed=42, n_seed_l1=1, is_agent_curriculum=True
    )
    return sorted({task.get_task_id() for task, _seed in tuples})


def split_pairs(
    n_train: int = 10, n_test: int = 10, seed: int = 42
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Build deterministic, disjoint (task_id, task_seed) pairs per task.

    For each task, seeds ``[seed, seed + n_train)`` form the train split and
    ``[seed + n_train, seed + n_train + n_test)`` form the test split, so the
    train and test seed ranges never overlap for any task.
    """
    train: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    for tid in l1_task_ids():
        seeds = list(range(seed, seed + n_train + n_test))
        train += [(tid, s) for s in seeds[:n_train]]
        test += [(tid, s) for s in seeds[n_train : n_train + n_test]]
    return train, test


def build_benchmark(pairs: list[tuple[str, int]], max_steps: int = 15) -> Benchmark:
    """Return a ``workarena_l1`` benchmark whose env_args_list is exactly ``pairs``."""
    bench = DEFAULT_BENCHMARKS["workarena_l1"]()
    bench.env_args_list = [
        EnvArgs(task_name=t, task_seed=s, max_steps=max_steps) for (t, s) in pairs
    ]
    return bench
