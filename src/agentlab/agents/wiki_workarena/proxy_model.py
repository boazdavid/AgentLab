"""Custom AgentLab ``ModelArgs`` that routes an agent through an
OpenAI-compatible litellm -> Bedrock proxy in text mode (no tool-calling).

Credentials are read from the environment (``OPENAI_BASE_URL`` +
``OPENAI_API_KEY``), optionally populated from a local ``.env``. Nothing here
prints or hardcodes secrets.
"""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel

# SNOW_*, OPENAI_*, AGENTLAB_EXP_ROOT, HF token. Missing file is fine.
load_dotenv(Path.home() / "Documents/GitHub/AgentLab/.env")


def _with_cache_control(messages):
    """Mark a cache breakpoint on the STABLE prefix (the system message).

    Anthropic caches the prefix up to and including the marked block, so the breakpoint
    MUST sit on stable content — marking the volatile tail (the per-step observation)
    would rewrite the cache every call and never read it (a net cost increase). The
    system message is the one block that is byte-identical across an episode's steps,
    so it is the safe breakpoint. Returns a shallow copy — never mutates AgentLab's
    message objects. (NOTE: AgentLab's system prompt is small; see module docstring for
    why real savings need the large stable content relocated into this cached block.)
    """
    if not messages:
        return messages
    msgs = [dict(m) for m in messages]

    def _mark(m):
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            blocks = [dict(b) if isinstance(b, dict) else b for b in content]
            for b in reversed(blocks):
                if isinstance(b, dict) and b.get("type") == "text":
                    b["cache_control"] = {"type": "ephemeral"}
                    break
            m["content"] = blocks

    # Cache ONLY the stable system head (system prompt + wiki index). Verified live to
    # read ~1624 tok/step at 0.1x with no write penalty.
    #
    # We deliberately do NOT put a breakpoint on the human message: AgentLab rebuilds it
    # each step (the AXTree observation is regenerated mid-prompt), so it is NOT
    # append-only — marking it wrote ~15k tok/step at the ~1.25x cache-WRITE premium and
    # almost never read them back (a net cost increase). The dominant per-step cost is the
    # AXTree observation, which changes every step and is inherently uncacheable.
    for m in msgs:
        if m.get("role") == "system":
            _mark(m)
            break
    return msgs


class CacheAwareChatModel(ChatModel):
    """ChatModel that enables Bedrock/Anthropic prompt caching through the litellm proxy.

    Two things are required and BOTH are done here (verified against the proxy):
    - a ``cache_control`` breakpoint on the prompt's stable prefix, and
    - a stable per-episode ``user`` id so the proxy routes every call in an episode to
      the SAME backend (without it the load balancer scatters calls and cache reads
      never hit — which would only add the write premium).

    The ``user`` is unique per model instance (i.e. per episode, since AgentLab builds
    a fresh agent+model per task), giving intra-episode cache reuse while still spreading
    episodes across backends for parallelism.
    """

    def __init__(self, *args, cache_user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_user = cache_user or ("wa-" + uuid.uuid4().hex[:16])
        _orig_create = self.client.chat.completions.create

        def _wrapped(**kw):
            if kw.get("messages"):
                kw["messages"] = _with_cache_control(kw["messages"])
            kw.setdefault("user", self._cache_user)  # sticky routing -> cache reads hit
            resp = _orig_create(**kw)
            log_path = os.environ.get("WA_CACHE_LOG")
            if log_path:  # optional diagnostics — off unless WA_CACHE_LOG is set
                try:
                    u = resp.usage
                    with open(log_path, "a") as f:
                        f.write(
                            f"{self._cache_user} prompt={u.prompt_tokens} "
                            f"read={getattr(u, 'cache_read_input_tokens', 0)} "
                            f"create={getattr(u, 'cache_creation_input_tokens', 0)}\n"
                        )
                except Exception:
                    pass
            return resp

        self.client.chat.completions.create = _wrapped


@dataclass
class ProxyModelArgs(BaseModelArgs):
    """litellm->Bedrock Claude via an OpenAI-compatible proxy. Text-mode (no tool-calling).

    Prompt caching is on by default (per-episode sticky routing); set WA_CACHE=0 to disable.
    """

    def make_model(self):
        model_cls = ChatModel if os.environ.get("WA_CACHE") == "0" else CacheAwareChatModel
        return model_cls(
            model_name=self.model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            client_class=OpenAI,
            client_args={"base_url": os.environ["OPENAI_BASE_URL"]},  # ".../v1"
        )


# Model id as the OpenAI SDK must send it to the proxy — NO "openai/" prefix (that is a
# litellm client-side routing convention; sending it here yields 403 model-access-denied).
# Override per run with WA_AGENT_MODEL (must be a model id the proxy team is allowed to access,
# e.g. claude-sonnet-4-5-20250929, claude-opus-4-8, aws/claude-sonnet-5).
DEFAULT_AGENT_MODEL = "claude-sonnet-4-5-20250929"

PROXY_MODEL_ARGS = ProxyModelArgs(
    model_name=os.environ.get("WA_AGENT_MODEL", DEFAULT_AGENT_MODEL),
    max_total_tokens=200_000,
    max_input_tokens=180_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=False,
)
