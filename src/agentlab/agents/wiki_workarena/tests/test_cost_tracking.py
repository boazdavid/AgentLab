"""Smoke tests for cost tracking on the proxy path (no network).

Guards the fix that made ``CacheAwareChatModel`` bill cached input tokens at the
cache-read rate and gave the proxy model real prices (previously ``cum_cost`` was
always 0 because ``make_model`` never passed a ``pricing_func``).

All tests construct the model with a FAKE OpenAI-compatible client, so nothing hits
the network — the real ``CacheAwareChatModel`` wrapper and ``__call__`` cost logic run.
"""

import types

import pytest

from agentlab.llm import tracking
from agentlab.agents.wiki_workarena.proxy_model import (
    CacheAwareChatModel,
    ProxyModelArgs,
    _PROXY_RATES,
)

IN_COST, OUT_COST = 1.52e-6, 7.60e-6  # $/token; matches _PROXY_RATES["aws/claude-sonnet-5"]
READ_F = tracking.ANTHROPIC_CACHE_PRICING_FACTOR["cache_read_tokens"]   # 0.1
WRITE_F = tracking.ANTHROPIC_CACHE_PRICING_FACTOR["cache_write_tokens"]  # 1.25


def _fake_client_factory(prompt_tokens, completion_tokens, cached=0, written=0):
    """Build a client_class whose chat.completions.create returns a fixed usage."""

    def create(**kw):
        usage = types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
            cache_creation_input_tokens=written,
        )
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content="ok", log_probs=None)
        )
        return types.SimpleNamespace(usage=usage, choices=[choice])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    return FakeClient


def _make(fake_client, prices=True):
    pricing = (
        (lambda: {"aws/claude-sonnet-5": {"prompt": IN_COST, "completion": OUT_COST}})
        if prices
        else None
    )
    return CacheAwareChatModel(
        model_name="aws/claude-sonnet-5",
        api_key="k",
        temperature=0.1,
        max_tokens=2000,
        client_class=fake_client,
        pricing_func=pricing,
    )


def _run(model):
    with tracking.set_tracker() as t:
        model([{"role": "system", "content": "s"}, {"role": "user", "content": "h"}])
        return dict(t.stats)


def test_cache_aware_cost_reads_and_writes():
    """Cached reads billed at 0.1x, writes at 1.25x, rest at 1x. prompt_tokens includes both."""
    model = _make(_fake_client_factory(10_000, 500, cached=6_000, written=200))
    stats = _run(model)
    regular = 10_000 - 6_000 - 200
    expected = (
        regular * IN_COST
        + 6_000 * IN_COST * READ_F
        + 200 * IN_COST * WRITE_F
        + 500 * OUT_COST
    )
    assert stats["cost"] == pytest.approx(expected, rel=1e-9)
    # token totals are untouched by the correction delta
    assert stats["input_tokens"] == 10_000
    assert stats["output_tokens"] == 500
    # and it must be cheaper than the old cache-blind flat cost
    flat = 10_000 * IN_COST + 500 * OUT_COST
    assert stats["cost"] < flat


def test_no_cache_equals_flat_cost():
    """With nothing cached, the correction is a no-op: cost == flat input/output cost."""
    model = _make(_fake_client_factory(8_000, 300, cached=0, written=0))
    stats = _run(model)
    flat = 8_000 * IN_COST + 300 * OUT_COST
    assert stats["cost"] == pytest.approx(flat, rel=1e-9)


def test_reads_via_anthropic_field_fallback():
    """Proxy may expose reads as usage.cache_read_input_tokens (no prompt_tokens_details)."""

    def create(**kw):
        usage = types.SimpleNamespace(
            prompt_tokens=5_000,
            completion_tokens=100,
            cache_read_input_tokens=4_000,  # Anthropic-style field
        )
        return types.SimpleNamespace(
            usage=usage,
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", log_probs=None))],
        )

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    model = _make(FakeClient)
    stats = _run(model)
    expected = (5_000 - 4_000) * IN_COST + 4_000 * IN_COST * READ_F + 100 * OUT_COST
    assert stats["cost"] == pytest.approx(expected, rel=1e-9)


def test_zero_prices_no_crash():
    """No pricing_func -> prices 0 -> cost 0, and the correction delta stays a no-op."""
    model = _make(_fake_client_factory(1_000, 50, cached=500), prices=False)
    assert model.input_cost == 0.0
    stats = _run(model)
    assert stats["cost"] == 0.0


def test_make_model_sets_prices_from_table(monkeypatch):
    """make_model must give aws/claude-sonnet-5 non-zero prices from _PROXY_RATES."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("WA_INPUT_COST", raising=False)
    monkeypatch.delenv("WA_OUTPUT_COST", raising=False)
    args = ProxyModelArgs(
        model_name="aws/claude-sonnet-5",
        max_total_tokens=1000, max_input_tokens=900, max_new_tokens=10,
        temperature=0.1, vision_support=False,
    )
    model = args.make_model()
    assert model.input_cost == _PROXY_RATES["aws/claude-sonnet-5"][0]
    assert model.output_cost == _PROXY_RATES["aws/claude-sonnet-5"][1]
    assert model.input_cost > 0


def test_make_model_env_overrides_prices(monkeypatch):
    """WA_INPUT_COST / WA_OUTPUT_COST override the table for any model id."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("WA_INPUT_COST", "9.9e-6")
    monkeypatch.setenv("WA_OUTPUT_COST", "3.3e-5")
    args = ProxyModelArgs(
        model_name="some/unlisted-model",
        max_total_tokens=1000, max_input_tokens=900, max_new_tokens=10,
        temperature=0.1, vision_support=False,
    )
    model = args.make_model()
    assert model.input_cost == pytest.approx(9.9e-6)
    assert model.output_cost == pytest.approx(3.3e-5)


def test_make_model_unknown_model_no_env_is_zero(monkeypatch):
    """Unknown model + no env override -> prices default to 0 (base warns)."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("WA_INPUT_COST", raising=False)
    monkeypatch.delenv("WA_OUTPUT_COST", raising=False)
    args = ProxyModelArgs(
        model_name="some/unlisted-model",
        max_total_tokens=1000, max_input_tokens=900, max_new_tokens=10,
        temperature=0.1, vision_support=False,
    )
    model = args.make_model()
    assert model.input_cost == 0.0
    assert model.output_cost == 0.0
