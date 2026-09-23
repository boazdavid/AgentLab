"""Tests for the wiki retrieval custom actions + action-set wiring.

Two deliberate departures from the task brief (see task-8 report):

(1) The actions raise a built-in ``Exception`` (not a custom ``KnowledgeResult``
    class). BrowserGym runs a custom action by ``exec(code, globals)`` where
    ``globals`` holds only ``page``/``send_message_to_user``/... + builtins, so a
    module-level custom class name is NOT in scope at real runtime. A built-in is.

(2) The actions use stdlib ``urllib.request`` (imports live *inside* each function
    body so they survive the getsource->exec copy). ``respx`` mocks ``httpx`` and
    would never intercept urllib, so we mock ``urllib.request.urlopen`` directly.
"""

import io
import json
from unittest import mock

import pytest

from agentlab.agents.wiki_workarena.retrieval_actions import (
    query_memories,
    get_memories,
    WikiActionSetArgs,
)


def _fake_urlopen(payload: dict):
    """Return a fake urlopen that yields a file-like object json.load can read."""

    def _open(req, timeout=None):
        return io.BytesIO(json.dumps(payload).encode())

    return _open


def test_query_memories_raises_builtin_with_text():
    with mock.patch("urllib.request.urlopen", _fake_urlopen({"text": "INDEX ROWS..."})):
        with pytest.raises(Exception) as excinfo:
            query_memories("create incident")
    # Must be a plain built-in Exception (exec namespace has no custom classes).
    assert type(excinfo.value) is Exception
    assert "WIKI SEARCH RESULTS:" in str(excinfo.value)
    assert "INDEX ROWS..." in str(excinfo.value)


def test_get_memories_raises_builtin_with_text():
    with mock.patch("urllib.request.urlopen", _fake_urlopen({"text": "BODY OF ARTICLE"})):
        with pytest.raises(Exception) as excinfo:
            get_memories(["create-incident", "assign-to-group"])
    assert type(excinfo.value) is Exception
    assert "WIKI ARTICLES:" in str(excinfo.value)
    assert "BODY OF ARTICLE" in str(excinfo.value)


def test_query_memories_posts_expected_payload():
    captured = {}

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["ctype"] = req.get_header("Content-type")
        return io.BytesIO(json.dumps({"text": "ok"}).encode())

    with mock.patch("urllib.request.urlopen", _open):
        with pytest.raises(Exception):
            query_memories("how to create an incident")

    assert captured["url"].endswith("/query_memories")
    assert json.loads(captured["body"]) == {"q": "how to create an incident"}
    assert captured["ctype"] == "application/json"


def test_get_memories_posts_memory_slugs_payload():
    captured = {}

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return io.BytesIO(json.dumps({"text": "ok"}).encode())

    with mock.patch("urllib.request.urlopen", _open):
        with pytest.raises(Exception):
            get_memories(["a", "b"])

    assert captured["url"].endswith("/get_memories")
    assert json.loads(captured["body"]) == {"memory_slugs": ["a", "b"]}


def test_knowledge_url_env_override(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_URL", "http://example.test:9000")
    captured = {}

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        return io.BytesIO(json.dumps({"text": "ok"}).encode())

    with mock.patch("urllib.request.urlopen", _open):
        with pytest.raises(Exception):
            query_memories("x")

    assert captured["url"] == "http://example.test:9000/query_memories"


def test_action_set_includes_custom_source():
    aset = WikiActionSetArgs(subsets=("workarena", "custom")).make_action_set()
    includes = aset.python_includes
    # getsource must have copied both action defs into the executable preamble.
    assert "def query_memories" in includes
    assert "def get_memories" in includes
    # Both actions are registered in the action set.
    assert "query_memories" in aset.action_set
    assert "get_memories" in aset.action_set


def test_action_set_subset_only_get_memories():
    """Index-batch arm exposes only get_memories, never query_memories."""
    aset = WikiActionSetArgs(
        action_names=("get_memories",), subsets=("workarena", "custom")
    ).make_action_set()
    includes = aset.python_includes
    assert "def get_memories" in includes
    assert "def query_memories" not in includes
    assert "get_memories" in aset.action_set
    assert "query_memories" not in aset.action_set


def test_action_set_default_keeps_both():
    """Default action_names stays backward compatible (both actions present)."""
    aset = WikiActionSetArgs(subsets=("workarena", "custom")).make_action_set()
    assert "def query_memories" in aset.python_includes
    assert "def get_memories" in aset.python_includes


def test_action_set_codegen_for_query():
    aset = WikiActionSetArgs(subsets=("workarena", "custom")).make_action_set()
    code = aset.to_python_code('query_memories("create incident")')
    assert "def query_memories" in code
    assert 'query_memories(\'create incident\')' in code or 'query_memories("create incident")' in code
