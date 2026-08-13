from agentlab.agents.wiki_workarena.splits import l1_task_ids, split_pairs, build_benchmark


def test_all_33_tasks():
    assert len(set(l1_task_ids())) == 33


def test_train_test_seeds_disjoint():
    train, test = split_pairs(n_train=10, n_test=10, seed=42)
    for tid in l1_task_ids():
        tr = {s for (t, s) in train if t == tid}
        te = {s for (t, s) in test if t == tid}
        assert tr and te and tr.isdisjoint(te)


def test_build_benchmark_uses_pairs():
    train, _ = split_pairs(2, 2, 42)
    bench = build_benchmark(train, max_steps=15)
    got = {(e.task_name, e.task_seed) for e in bench.env_args_list}
    assert got == set(train)
