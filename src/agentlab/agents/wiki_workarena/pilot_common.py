"""Shared helpers for the WorkArena wiki-knowledge PILOT (index-batch arm).

- ``render_index_preamble(wiki_dir)`` — build the browser-framed knowledge
  preamble by injecting a built wiki's ``index.md`` catalog into
  ``preamble_index_batch.txt`` at the ``{{INDEX_MD}}`` placeholder.
- ``select_pairs(tasks, n_train, n_test)`` — filter the deterministic
  ``split_pairs`` output down to task_ids matching any of the given substrings.
"""

from pathlib import Path

from agentlab.agents.wiki_workarena.splits import l1_task_ids, split_pairs

_PREAMBLE_PATH = Path(__file__).with_name("preamble_index_batch.txt")
_INDEX_PLACEHOLDER = "{{INDEX_MD}}"


def render_index_preamble(wiki_dir) -> str:
    """Return the index-batch preamble with the wiki's ``index.md`` injected.

    Reads ``preamble_index_batch.txt`` (next to this module) and the built wiki's
    catalog at ``<wiki_dir>/index.md``, then substitutes ``{{INDEX_MD}}`` with the
    catalog contents.
    """
    preamble = _PREAMBLE_PATH.read_text(encoding="utf-8")
    index_md = (Path(wiki_dir) / "index.md").read_text(encoding="utf-8")
    return preamble.replace(_INDEX_PLACEHOLDER, index_md)


def select_pairs(tasks, n_train: int, n_test: int):
    """Filter ``split_pairs(n_train, n_test)`` to task_ids matching any substring.

    ``tasks`` is a list of task-id substrings (e.g. ``["create-incident"]``). A
    (task_id, seed) pair is kept if its task_id contains any of the substrings.
    Returns ``(train_pairs, test_pairs)`` preserving split_pairs' ordering.
    """
    subs = [s for s in tasks if s]
    train, test = split_pairs(n_train=n_train, n_test=n_test)

    def _keep(pair):
        tid = pair[0]
        return any(sub in tid for sub in subs)

    return [p for p in train if _keep(p)], [p for p in test if _keep(p)]


# The 6 pilot task types, as substrings that resolve against l1_task_ids().
# Resolved ids (2026-08): workarena.servicenow.{create-incident,
# create-change-request, create-user, filter-incident-list, sort-user-list,
# order-developer-laptop}.
PILOT_TASKS = [
    "create-incident",
    "create-change",
    "create-user",
    "filter-incident",
    "sort-user",
    "order-developer-laptop",
]


def resolve_pilot_ids(tasks=None):
    """Return the concrete l1 task_ids that ``tasks`` substrings resolve to."""
    subs = tasks or PILOT_TASKS
    return sorted(
        tid for tid in l1_task_ids() if any(sub in tid for sub in subs)
    )
