"""Deterministic train/test seed split for WorkArena + benchmark builder.

Parametrized by ``tier`` ("l1" or "l2"); L1 is the default and its behavior is
unchanged from the original L1-only version.

Interfaces consumed by later tasks (keep names/shapes stable):
- ``task_ids(tier="l1") -> list[str]`` — the WorkArena atomic task ids for a tier.
- ``l1_task_ids()`` / ``l2_task_ids()`` — convenience wrappers over ``task_ids``.
- ``split_pairs(n_train=10, n_test=10, seed=42, tier="l1")`` — disjoint
  (task_id, seed) pairs, generated over ``task_ids(tier)``.
- ``build_benchmark(pairs, max_steps=None, tier="l1")`` — a benchmark whose
  ``env_args_list`` is exactly ``pairs``. Default ``max_steps`` is 15 for l1
  and 50 for l2.
"""

from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
from browsergym.experiments.loop import EnvArgs

# Per-tier benchmark template key in DEFAULT_BENCHMARKS and default max_steps.
_TIER_BENCHMARK = {
    "l1": "workarena_l1",
    "l2": "workarena_l2_agent_curriculum_eval",
}
_TIER_MAX_STEPS = {"l1": 15, "l2": 50}


def task_ids(tier: str = "l1") -> list[str]:
    """Return the sorted, distinct WorkArena task ids for ``tier`` ("l1"/"l2").

    Enumerates task classes locally (no live ServiceNow instance required). The
    curriculum seed count does not affect the distinct task-id set, so we always
    request ``n_seed_l1=10``.
    """
    from browsergym.workarena import get_all_tasks_agents

    tuples = get_all_tasks_agents(
        filter=tier, meta_seed=42, n_seed_l1=10, is_agent_curriculum=True
    )
    return sorted({task.get_task_id() for task, _seed in tuples})


def l1_task_ids() -> list[str]:
    """Return the sorted, distinct WorkArena-L1 atomic task ids."""
    return task_ids("l1")


def l2_task_ids() -> list[str]:
    """Return the sorted, distinct WorkArena-L2 task ids."""
    return task_ids("l2")


def split_pairs(
    n_train: int = 10, n_test: int = 10, seed: int = 42, tier: str = "l1"
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Build deterministic, disjoint (task_id, task_seed) pairs per task.

    For each task in ``task_ids(tier)``, seeds ``[seed, seed + n_train)`` form the
    train split and ``[seed + n_train, seed + n_train + n_test)`` form the test
    split, so the train and test seed ranges never overlap for any task.
    """
    train: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    for tid in task_ids(tier):
        seeds = list(range(seed, seed + n_train + n_test))
        train += [(tid, s) for s in seeds[:n_train]]
        test += [(tid, s) for s in seeds[n_train : n_train + n_test]]
    # Order the TEST pairs seed-major (one seed of every task type first, then
    # repeat with the next seed) so a partial/interrupted run covers all task types
    # evenly instead of finishing early-alphabet tasks first — partial scores stay
    # representative. Scoring pairs by (task, seed), so order does not affect results.
    # Train is left task-major so it can't silently change a --no-shuffle build's ingest order.
    test.sort(key=lambda p: (p[1], p[0]))
    return train, test


def build_benchmark(
    pairs: list[tuple[str, int]], max_steps: int | None = None, tier: str = "l1"
) -> Benchmark:
    """Return a WorkArena benchmark whose env_args_list is exactly ``pairs``.

    The benchmark template is chosen by ``tier`` ("l1" -> ``workarena_l1``, "l2" ->
    ``workarena_l2_agent_curriculum_eval``). When ``max_steps`` is ``None`` it
    defaults to 15 for l1 and 50 for l2.
    """
    if max_steps is None:
        max_steps = _TIER_MAX_STEPS[tier]
    bench = DEFAULT_BENCHMARKS[_TIER_BENCHMARK[tier]]()
    bench.env_args_list = [
        EnvArgs(task_name=t, task_seed=s, max_steps=max_steps) for (t, s) in pairs
    ]
    return bench
