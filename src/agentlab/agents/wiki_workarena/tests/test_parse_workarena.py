"""Offline unit tests for the pure mapping layer of parse_workarena.

These tests exercise ``episode_to_trajectory_dict`` with a FABRICATED list of
step dicts (shaped like ``load_episode`` output). They do NOT touch pickles,
browsergym, or the network, so they run fast and fully offline.
"""

import json

from agentlab.agents.wiki_workarena.parse_workarena import episode_to_trajectory_dict


def _fabricated_steps():
    """A hand-built episode covering every mapping branch.

    Shapes mirror /tmp/wa_traj.json: each step carries the raw ``think`` with an
    ``<action>...</action>`` wrapper, the ``action`` string, ``action_name``,
    positional ``action_args`` ({"args": [...]}), plus obs fields. The
    success/error/retrieval outcome of step i is determined by step i+1's
    ``last_action_error``.
    """
    return [
        # 0: click that succeeds (next step has no error)
        {
            "think": "I should open the app navigator.\n\n<action>\nclick('79')\n</action>",
            "action": "click('79')",
            "action_name": "click",
            "action_args": {"args": ["79"]},
            "url": "https://example.service-now.com/home",
            "axtree_head": "RootWebArea 'Home'",
            "last_action_error": None,
            "reward": 0,
            "terminated": False,
        },
        # 1: fill whose NEXT step carries a real (non-WIKI) error -> failure
        {
            "think": "\n\n<action>\nfill('242', 'Configuration')\n</action>",
            "action": "fill('242', 'Configuration')",
            "action_name": "fill",
            "action_args": {"args": ["242", "Configuration"]},
            "url": "https://example.service-now.com/home",
            "axtree_head": "RootWebArea 'Dashboard'",
            "last_action_error": None,
            "reward": 0,
            "terminated": False,
        },
        # 2: query_articles whose NEXT step carries a WIKI SEARCH RESULTS payload
        {
            "think": "Let me consult the wiki.\n\n<action>\nquery_articles('how to navigate modules')\n</action>",
            "action": "query_articles('how to navigate modules')",
            "action_name": "query_articles",
            "action_args": {"args": ["how to navigate modules"]},
            "url": "https://example.service-now.com/home",
            # this step's OWN error is the real error produced by step 1's fill
            "last_action_error": "TimeoutError: locator '242' not found",
            "reward": 0,
            "terminated": False,
        },
        # 3: click after wiki results; its own error is the WIKI payload from step 2
        {
            "think": "<action>\nclick('2870')\n</action>",
            "action": "click('2870')",
            "action_name": "click",
            "action_args": {"args": ["2870"]},
            "url": "https://example.service-now.com/home",
            "last_action_error": "WIKI SEARCH RESULTS:\nconcepts/navigate.md — how to navigate the app",
            "reward": 0,
            "terminated": False,
        },
        # 4: terminal step, no action
        {
            "think": "",
            "action": None,
            "action_name": None,
            "action_args": {},
            "url": "https://example.service-now.com/list.do",
            "axtree_head": "RootWebArea 'Classic'",
            "last_action_error": None,
            "reward": 1,
            "terminated": True,
        },
    ]


def _traj():
    return episode_to_trajectory_dict(
        goal="Navigate to the ESX Servers module.",
        steps=_fabricated_steps(),
        file_path="episodes/task_43",
    )


def test_top_level_schema_fields():
    t = _traj()
    assert set(t.keys()) == {"file_path", "system_prompt", "tools", "messages"}
    assert t["file_path"] == "episodes/task_43"
    assert t["system_prompt"] == ""
    assert t["tools"] == []
    assert isinstance(t["messages"], list)


def test_first_message_is_task_goal():
    m0 = _traj()["messages"][0]
    assert m0["role"] == "user"
    assert m0["content"] == "TASK GOAL: Navigate to the ESX Servers module."
    assert m0["tool_calls"] == []


def test_click_step_named_args_and_reasoning():
    msgs = _traj()["messages"]
    # message[1] is the assistant for step 0 (click)
    a = msgs[1]
    assert a["role"] == "assistant"
    assert a["content"] == "I should open the app navigator."
    assert "<action>" not in a["content"]
    assert len(a["tool_calls"]) == 1
    tc = a["tool_calls"][0]
    assert tc["tool_name"] == "click"
    assert tc["args"] == {"bid": "79"}


def test_fill_step_named_args():
    tc = _find_first_toolcall(_traj(), "fill")
    assert tc["args"] == {"bid": "242", "value": "Configuration"}


def test_real_error_marks_failure():
    # step 1 (fill) fails because step 2's last_action_error is a real error
    tc = _find_first_toolcall(_traj(), "fill")
    assert tc["success"] is False
    assert tc["error_text"] == "TimeoutError: locator '242' not found"
    assert tc["response"] is None


def test_retrieval_delivers_wiki_text_as_response():
    # step 2 (query_articles): step 3's error starts with WIKI SEARCH RESULTS
    tc = _find_first_toolcall(_traj(), "query_articles")
    assert tc["success"] is True
    assert tc["error_text"] is None
    assert tc["response"] is not None
    assert "concepts/navigate.md" in tc["response"]
    # leading marker line may be stripped
    assert "how to navigate the app" in tc["response"]


def test_empty_reasoning_placeholder():
    # step 3's think is only the <action> wrapper -> placeholder reasoning
    a = _assistant_for_toolcall(_traj(), "click", occurrence=1)
    assert a["content"] == "(no reasoning text)"


def test_observation_messages_present():
    msgs = _traj()["messages"]
    obs = [m for m in msgs if m["role"] == "user" and m["content"].startswith("[observation]")]
    assert obs
    assert any("url=https://example.service-now.com/home" in m["content"] for m in obs)
    assert all(m["tool_calls"] == [] for m in obs)


def test_terminal_end_message_present():
    msgs = _traj()["messages"]
    ends = [m for m in msgs if m["role"] == "assistant" and m["content"].startswith("(episode end")]
    assert len(ends) == 1
    assert "reward=1" in ends[0]["content"]
    assert "terminated=True" in ends[0]["content"]
    assert ends[0]["tool_calls"] == []


def test_round_trips_through_json():
    t = _traj()
    assert json.loads(json.dumps(t)) == t


# --- helpers ---------------------------------------------------------------

def _find_first_toolcall(traj, tool_name):
    for m in traj["messages"]:
        for tc in m.get("tool_calls", []):
            if tc["tool_name"] == tool_name:
                return tc
    raise AssertionError(f"no tool_call named {tool_name}")


def _assistant_for_toolcall(traj, tool_name, occurrence=0):
    seen = 0
    for m in traj["messages"]:
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls", []):
            if tc["tool_name"] == tool_name:
                if seen == occurrence:
                    return m
                seen += 1
    raise AssertionError(f"no assistant msg #{occurrence} for {tool_name}")
