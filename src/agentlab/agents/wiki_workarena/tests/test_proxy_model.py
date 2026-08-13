import os

import pytest

from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS


def test_proxy_roundtrips_text():
    if not os.getenv("OPENAI_BASE_URL"):
        pytest.skip("OPENAI_BASE_URL unset")
    model = PROXY_MODEL_ARGS.make_model()
    out = model([{"role": "user", "content": "Reply with the single word: pong"}])
    assert "pong" in out["content"].lower()
