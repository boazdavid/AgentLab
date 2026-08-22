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
        # 0: click that succeeds. Its RESULTING observation is step 1's state
        # (url=/dashboard, axtree "RootWebArea 'Dashboard'").
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
        # 1: fill whose NEXT step carries a real (non-WIKI) error -> failure.
        {
            "think": "\n\n<action>\nfill('242', 'Configuration')\n</action>",
            "action": "fill('242', 'Configuration')",
            "action_name": "fill",
            "action_args": {"args": ["242", "Configuration"]},
            "url": "https://example.service-now.com/dashboard",
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
            "url": "https://example.service-now.com/search",
            "axtree_head": "RootWebArea 'Search'",
            # this step's OWN error is the real error produced by step 1's fill
            "last_action_error": "TimeoutError: locator '242' not found",
            "reward": 0,
            "terminated": False,
        },
        # 3: click after wiki results; its own error is the WIKI payload from step 2.
        # Its RESULTING observation is step 4's state (url=/list.do).
        {
            "think": "<action>\nclick('2870')\n</action>",
            "action": "click('2870')",
            "action_name": "click",
            "action_args": {"args": ["2870"]},
            "url": "https://example.service-now.com/results",
            "axtree_head": "RootWebArea 'Results'",
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


def test_first_action_is_preceded_by_observation():
    # KEY FIX: each step's OWN observation is emitted as a user message BEFORE
    # that step's assistant action — including the very first action.
    msgs = _traj()["messages"]
    # message[1] is the observation the agent SAW to decide step 0's action.
    m1 = msgs[1]
    assert m1["role"] == "user"
    assert m1["content"].startswith("[observation]")
    assert "url=https://example.service-now.com/home" in m1["content"]
    assert "RootWebArea 'Home'" in m1["content"]
    assert m1["tool_calls"] == []
    # message[2] is the first assistant action, immediately AFTER the observation.
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["tool_name"] == "click"


def test_click_step_named_args_and_reasoning():
    msgs = _traj()["messages"]
    # message[2] is the assistant for step 0 (click); message[1] is its observation.
    a = msgs[2]
    assert a["role"] == "assistant"
    assert a["content"] == "I should open the app navigator."
    assert "<action>" not in a["content"]
    assert len(a["tool_calls"]) == 1
    tc = a["tool_calls"][0]
    assert tc["tool_name"] == "click"
    assert tc["args"] == {"bid": "79"}
    # regular browser action success: response is None — the observation the
    # agent saw lives in the PRECEDING user message (message[1]), not here.
    assert tc["success"] is True
    assert tc["error_text"] is None
    assert tc["response"] is None
    assert "url=https://example.service-now.com/home" in msgs[1]["content"]


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


def test_observation_precedes_every_assistant_message():
    # observe -> act: every assistant message is immediately preceded by an
    # [observation] user message carrying that step's OWN page. There is one
    # observation per step (5 steps) plus the single TASK GOAL user message.
    msgs = _traj()["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    obs_msgs = [m for m in msgs if m["content"].startswith("[observation]")]
    assert len(user_msgs) == 6  # goal + 5 observations
    assert len(obs_msgs) == 5
    for idx, m in enumerate(msgs):
        if m["role"] == "assistant":
            prev = msgs[idx - 1]
            assert prev["role"] == "user"
            assert prev["content"].startswith("[observation]")


def test_terminal_step_observation_then_end_message():
    msgs = _traj()["messages"]
    ends = [
        (idx, m)
        for idx, m in enumerate(msgs)
        if m["role"] == "assistant" and m["content"].startswith("(episode end")
    ]
    assert len(ends) == 1
    idx, end = ends[0]
    assert "reward=1" in end["content"]
    assert "terminated=True" in end["content"]
    assert end["tool_calls"] == []
    # the episode-end message is the LAST message.
    assert idx == len(msgs) - 1
    # the terminal step's OWN final page precedes the end message.
    obs = msgs[idx - 1]
    assert obs["role"] == "user"
    assert obs["content"].startswith("[observation]")
    assert "url=https://example.service-now.com/list.do" in obs["content"]


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
