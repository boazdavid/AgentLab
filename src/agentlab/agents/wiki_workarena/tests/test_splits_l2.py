"""Tests for WorkArena-L2 support in the tier-parametrized splits helpers."""

from agentlab.agents.wiki_workarena.splits import (
    build_benchmark,
    l1_task_ids,
    l2_task_ids,
    split_pairs,
)


def test_l2_task_ids_count_and_shape():
    ids = l2_task_ids()
    assert len(set(ids)) == 152
    assert all("-l2" in i for i in ids)
    assert all(i.startswith("workarena.servicenow.") for i in ids)


def test_l2_split_pairs_disjoint_seeds():
    train, test = split_pairs(n_train=3, n_test=2, tier="l2")
    tid = l2_task_ids()[0]
    tr = {s for (t, s) in train if t == tid}
    te = {s for (t, s) in test if t == tid}
    assert len(tr) == 3
    assert len(te) == 2
    assert tr.isdisjoint(te)


def test_l2_build_benchmark_pairs_and_max_steps():
    train, _ = split_pairs(n_train=2, n_test=2, tier="l2")
    bench = build_benchmark(train, tier="l2")
    got = {(e.task_name, e.task_seed) for e in bench.env_args_list}
    assert got == set(train)
    assert all(e.max_steps == 50 for e in bench.env_args_list)


def test_l1_build_benchmark_defaults_to_15():
    train, _ = split_pairs(n_train=2, n_test=2, tier="l1")
    bench = build_benchmark(train, tier="l1")
    assert all(e.max_steps == 15 for e in bench.env_args_list)


def test_l1_default_tier_unchanged():
    # l1 remains the default for both task enumeration and splits.
    assert split_pairs(2, 2) == split_pairs(2, 2, tier="l1")
    assert set(l1_task_ids()) != set(l2_task_ids())
