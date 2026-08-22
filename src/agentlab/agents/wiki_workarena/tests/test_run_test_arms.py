"""Test the set_benchmark wiring for the retrieval (index-batch) test arm.

Asserts (without .run()) that after ``agent.set_benchmark(bench)`` the retrieval
arm's ``agent.flags.action.action_set`` is a WikiActionSetArgs whose action set
exposes get_articles and NOT query_articles.
"""

from agentlab.agents.wiki_workarena.cached_agent import CachedSystemAgentArgs
from agentlab.agents.wiki_workarena.retrieval_actions import WikiActionSetArgs
from agentlab.agents.wiki_workarena.run_test_arms import build_arm


def _fake_wiki(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text("- [x](concepts/x.md) — x\n", encoding="utf-8")
    return wiki


def test_retrieval_arm_set_benchmark_wiring(tmp_path):
    pairs = [("workarena.servicenow.create-incident", 42)]
    agent_args, bench = build_arm("retrieval", pairs, wiki_dir=_fake_wiki(tmp_path))

    # Emulate AgentLab's set_benchmark (deepcopies bench.high_level_action_set_args).
    agent_args.set_benchmark(bench, demo_mode=False)

    # The retrieval arm now uses CachedSystemAgentArgs so the catalog rides in the
    # cached system prompt, not the volatile human message.
    assert isinstance(agent_args, CachedSystemAgentArgs)

    action_set = agent_args.flags.action.action_set
    assert isinstance(action_set, WikiActionSetArgs)
    assert action_set.action_names == ("get_articles",)

    aset = action_set.make_action_set()
    assert "def get_articles" in aset.python_includes
    assert "def query_articles" not in aset.python_includes

    # The injected catalog now lives in cached_system_suffix, NOT extra_instructions.
    assert "{{INDEX_MD}}" not in agent_args.cached_system_suffix
    assert "concepts/x.md" in agent_args.cached_system_suffix
    assert "concepts/x.md" not in (agent_args.flags.extra_instructions or "")


def test_control_arm_uses_stock_action_set(tmp_path):
    pairs = [("workarena.servicenow.create-incident", 42)]
    agent_args, bench = build_arm("control", pairs)
    agent_args.set_benchmark(bench, demo_mode=False)
    assert not isinstance(agent_args.flags.action.action_set, WikiActionSetArgs)
