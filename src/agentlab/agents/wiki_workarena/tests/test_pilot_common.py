"""Tests for the pilot helpers: index-preamble rendering and pair selection."""

from agentlab.agents.wiki_workarena.pilot_common import (render_index_preamble,
                                                         resolve_pilot_ids,
                                                         select_pairs)


def test_render_index_preamble_injects_catalog(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text(
        "- [Create an incident](memories/create-incident.md) — how to create one\n",
        encoding="utf-8",
    )
    out = render_index_preamble(wiki)
    assert "{{INDEX_MD}}" not in out  # placeholder consumed
    assert "memories/create-incident.md" in out  # fake catalog present
    assert "get_memories" in out  # index-batch action mentioned
    assert "query_memories" not in out  # no search action in this arm


def test_select_pairs_single_task():
    train, test = select_pairs(["create-incident"], n_train=3, n_test=3)
    assert len(train) == 3
    assert len(test) == 3
    tids = {p[0] for p in train + test}
    assert tids == {"workarena.servicenow.create-incident"}


def test_select_pairs_train_test_disjoint_seeds():
    train, test = select_pairs(["create-incident"], n_train=3, n_test=3)
    train_seeds = {p[1] for p in train}
    test_seeds = {p[1] for p in test}
    assert train_seeds.isdisjoint(test_seeds)


def test_resolve_pilot_ids():
    ids = resolve_pilot_ids()
    assert ids == [
        "workarena.servicenow.create-change-request",
        "workarena.servicenow.create-incident",
        "workarena.servicenow.create-user",
        "workarena.servicenow.filter-incident-list",
        "workarena.servicenow.order-developer-laptop",
        "workarena.servicenow.sort-user-list",
    ]
