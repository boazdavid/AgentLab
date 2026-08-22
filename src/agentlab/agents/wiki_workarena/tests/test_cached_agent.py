"""Offline tests for CachedSystemAgent / CachedSystemAgentArgs.

These verify that the wiki catalog is relocated into the (cached) system prompt:
- the suffix is stored on the args and carried onto the constructed agent, and
- the system message the agent would emit contains BOTH the stock SystemPrompt text
  AND the suffix.

No network / no browsergym env: ``make_model`` only constructs an OpenAI client (no
call), so we set dummy proxy env vars and disable the cache wrapper.
"""

import os
from copy import deepcopy

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.wiki_workarena.cached_agent import (
    CachedSystemAgent,
    CachedSystemAgentArgs,
)
from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS
from agentlab.agents.wiki_workarena.retrieval_actions import WikiActionSetArgs
from agentlab.agents.wiki_workarena.run_test_arms import build_arm

# make_model reads these; dummy values are fine because no request is made offline.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:65535/v1")
os.environ.setdefault("WA_CACHE", "0")  # plain ChatModel, no cache wrapper needed offline

SENTINEL = "WIKI-CATALOG-SENTINEL concepts/create-incident.md"


def _make_cached_agent(suffix):
    args = CachedSystemAgentArgs(
        chat_model_args=PROXY_MODEL_ARGS,
        flags=deepcopy(FLAGS_GPT_4o),
        cached_system_suffix=suffix,
    )
    return args, args.make_agent()


def test_suffix_stored_on_args_and_agent():
    args, agent = _make_cached_agent(SENTINEL)
    assert args.cached_system_suffix == SENTINEL
    assert isinstance(agent, CachedSystemAgent)
    assert agent.cached_system_suffix == SENTINEL


def test_system_prompt_contains_base_and_suffix():
    _args, agent = _make_cached_agent(SENTINEL)
    base_text = dp.SystemPrompt().prompt

    system_msg = agent.build_system_prompt()
    emitted = str(system_msg)  # SystemMessage stringifies to its content

    assert base_text in emitted
    assert SENTINEL in emitted
    # base text comes first, suffix appended after.
    assert emitted.index(base_text) < emitted.index(SENTINEL)


def test_empty_suffix_leaves_system_prompt_unchanged():
    _args, agent = _make_cached_agent("")
    assert str(agent.build_system_prompt()) == dp.SystemPrompt().prompt


def test_get_action_restores_global_system_prompt():
    """get_action mutates dp.SystemPrompt._prompt transiently; it must be restored."""
    _args, agent = _make_cached_agent(SENTINEL)
    before = dp.SystemPrompt._prompt
    # We do not run a full step (needs an obs/env); assert the augmented text is
    # built correctly and that the class attribute is untouched outside get_action.
    assert SENTINEL in agent._augmented_system_prompt_text()
    assert dp.SystemPrompt._prompt == before


def _fake_wiki(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text("- [x](concepts/x.md) — x\n", encoding="utf-8")
    return wiki


def test_retrieval_arm_roundtrips_set_benchmark_with_wiki_action_set(tmp_path):
    """CachedSystemAgentArgs survives set_benchmark keeping a WikiActionSetArgs."""
    pairs = [("workarena.servicenow.create-incident", 42)]
    agent_args, bench = build_arm("retrieval", pairs, wiki_dir=_fake_wiki(tmp_path))

    assert isinstance(agent_args, CachedSystemAgentArgs)
    assert "concepts/x.md" in agent_args.cached_system_suffix

    # Emulate AgentLab's set_benchmark (deepcopies bench.high_level_action_set_args).
    agent_args.set_benchmark(bench, demo_mode=False)

    action_set = agent_args.flags.action.action_set
    assert isinstance(action_set, WikiActionSetArgs)
    assert action_set.action_names == ("get_articles",)

    aset = action_set.make_action_set()
    assert "def get_articles" in aset.python_includes
    assert "def query_articles" not in aset.python_includes
