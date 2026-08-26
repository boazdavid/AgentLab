import os

import pytest

from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS


def test_proxy_roundtrips_text():
    if not os.getenv("OPENAI_BASE_URL"):
        pytest.skip("OPENAI_BASE_URL unset")
    model = PROXY_MODEL_ARGS.make_model()
    out = model([{"role": "user", "content": "Reply with the single word: pong"}])
    assert "pong" in out["content"].lower()


def test_make_model_ok_with_header_auth_and_no_api_key(monkeypatch):
    """contextguru/Anthropic endpoint: auth via ANTHROPIC_CUSTOM_HEADERS, no OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://contextguru.example/anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-context-guru-token: cg_test_123")
    monkeypatch.setenv("WA_CACHE", "0")  # plain ChatModel, no cache wrapper
    from agentlab.agents.wiki_workarena.proxy_model import ProxyModelArgs

    args = ProxyModelArgs(
        model_name="claude-sonnet-4-5-20250929",
        max_total_tokens=1000, max_input_tokens=900, max_new_tokens=10,
        temperature=0.1, vision_support=False,
    )
    model = args.make_model()          # previously raised KeyError: 'OPENAI_API_KEY'
    assert model.client.api_key         # some non-empty placeholder key was supplied
