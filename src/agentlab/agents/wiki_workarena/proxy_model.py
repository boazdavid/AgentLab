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
from agentlab.llm import tracking

# SNOW_*, OPENAI_*, AGENTLAB_EXP_ROOT, HF token. Missing file is fine.
load_dotenv(Path.home() / "Documents/GitHub/AgentLab/.env")

# Use the OS trust store (e.g. macOS keychain) for TLS instead of certifi's bundle, so
# the proxy's corporate/internal CA is trusted — matching `curl`. Without this the
# openai/httpx client raises CERTIFICATE_VERIFY_FAILED ("unable to get local issuer")
# and every call surfaces as a generic "Connection error", while curl to the same URL
# succeeds. inject_into_ssl() patches the stdlib ssl module process-wide (covers Ray
# workers that import this module to build the model). No-op if truststore is absent.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass


# RITS default endpoint for moonshotai/Kimi-K2.7-Code (one deployment == one model).
# Override with RITS_BASE_URL to point at a different RITS deployment.
DEFAULT_RITS_BASE_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
    "moonshotai-kimi-k2-7/v1"
)


def _use_rits():
    """Route through RITS when WA_RITS=1 (explicit), or when both RITS_API_KEY and
    RITS_BASE_URL are set (implicit). RITS_API_KEY alone does NOT trigger it, so an
    unrelated key in .env never hijacks the default proxy path."""
    if os.environ.get("WA_RITS") == "1":
        return True
    return bool(os.environ.get("RITS_API_KEY") and os.environ.get("RITS_BASE_URL"))


def _llm_log_path():
    """``llm_calls.jsonl`` inside the current episode folder, so the per-call request/response
    log lands next to step_*.pkl.gz / summary_info.json. The folder is published by
    ExpArgs.run as AGENTLAB_CURRENT_EXP_DIR; returns None outside an episode (e.g. standalone
    LLM calls), in which case nothing is logged."""
    exp_dir = os.environ.get("AGENTLAB_CURRENT_EXP_DIR")
    if exp_dir:
        return os.path.join(exp_dir, "llm_calls.jsonl")
    return None


def _install_call_logger(model):
    """Append the COMPLETE request and response (one JSON line per LLM call) to the episode
    folder's ``llm_calls.jsonl``: ``{"request": <create kwargs: model, messages, params>,
    "response": <resp.model_dump()>}``. Wraps the (possibly already-wrapped) create() so it
    works in any cache mode. The destination is resolved lazily per call (the episode dir isn't
    known when the model is built); a call outside an episode simply doesn't write. json.dumps
    uses default=str so anything unusual is stringified rather than raising, and failures are
    swallowed so logging never breaks a run. NOTE: this records full prompts (incl. the AXTree
    observation), so llm_calls.jsonl can be large."""
    import json

    _orig_create = model.client.chat.completions.create

    def _logged(**kw):
        resp = _orig_create(**kw)
        try:
            path = _llm_log_path()
            if path:
                req = dict(kw)
                # `messages` is a Discussion (custom, non-list) whose elements are AIMessage
                # (a dict subclass). json can't serialize the Discussion itself, so expand it
                # to plain dicts to preserve structured role/content instead of a str() blob.
                msgs = req.get("messages")
                if msgs is not None:
                    try:
                        req["messages"] = [dict(m) if isinstance(m, dict) else m for m in msgs]
                    except Exception:
                        pass
                record = {
                    "request": req,
                    "response": resp.model_dump() if hasattr(resp, "model_dump") else resp,
                }
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass
        return resp

    model.client.chat.completions.create = _logged


def _custom_headers():
    """Parse ANTHROPIC_CUSTOM_HEADERS into a dict, or return {} if unset/empty.

    Uses Claude Code's format: one ``Name: Value`` header per line (e.g.
    ``x-context-guru-token: cg_live_xxx``). Blank lines and lines without a
    colon are ignored.
    """
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
    if not raw:
        return {}
    headers = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip()] = value.strip()
    return headers


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
        # Claude honors explicit cache_control breakpoints + its own cache-read discount.
        # OpenAI-shape models (e.g. azure/gpt-5.5) cache prompt prefixes automatically
        # server-side and REJECT cache_control blocks, so for them we skip the injection
        # and correct billing from usage.cached_tokens at OPENAI_CACHE_READ_FACTOR instead.
        m = (self.model_name or "").lower()
        self._is_anthropic = "claude" in m or "anthropic" in m
        # Cache-token counts from the most recent create() call; consumed by __call__ to
        # correct the (cache-blind) cost the base ChatModel records. Reset per __call__.
        self._last_cached_tokens = 0
        self._last_written_tokens = 0
        _orig_create = self.client.chat.completions.create

        def _wrapped(**kw):
            if kw.get("messages") and self._is_anthropic:
                kw["messages"] = _with_cache_control(kw["messages"])
            kw.setdefault("user", self._cache_user)  # sticky routing -> cache reads hit
            resp = _orig_create(**kw)
            u = getattr(resp, "usage", None)
            if u is not None:
                # OpenAI-compat surfaces cache reads at usage.prompt_tokens_details.cached_tokens;
                # the litellm->Bedrock proxy also passes through Anthropic's *_input_tokens fields.
                details = getattr(u, "prompt_tokens_details", None)
                self._last_cached_tokens = (
                    getattr(details, "cached_tokens", 0)
                    or getattr(u, "cache_read_input_tokens", 0)
                    or 0
                )
                self._last_written_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
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

    def __call__(self, messages, n_samples: int = 1, temperature: float = None):
        """Bill cached input tokens at the cache-read rate instead of the full input rate.

        The base ``ChatModel.__call__`` bills every ``prompt_token`` at the full input rate
        and records that cost to the tracker. On the OpenAI-compatible path ``prompt_tokens``
        INCLUDES the cached (and cache-creation) tokens, so cached reads get overbilled at
        1x rather than the 0.1x cache-read rate. Rather than duplicate the base's retry loop,
        we post a correction *delta* (0 tokens, adjusted cost) to the same tracker:
          - reads:  refund ``cached * input_cost * (1 - cache_read_factor)``  (0.1x effective)
          - writes: add the cache-write premium ``written * input_cost * (cache_write_factor - 1)``
                    (+0.25x), assuming cache-creation tokens are counted within ``prompt_tokens``,
                    per the OpenAI-compat usage contract.
        Reads are the dominant, verified case; writes are tiny here (only the small system head
        is cached). No-op when prices are 0 or nothing was cached.
        """
        self._last_cached_tokens = 0
        self._last_written_tokens = 0
        res = super().__call__(messages, n_samples=n_samples, temperature=temperature)
        if self.input_cost:
            if self._is_anthropic:
                f = tracking.ANTHROPIC_CACHE_PRICING_FACTOR
                read_factor, write_factor = f["cache_read_tokens"], f["cache_write_tokens"]
            else:
                # OpenAI-shape: cached input bills at OPENAI_CACHE_READ_FACTOR x the input rate,
                # no cache-write premium (caching is automatic, server-side).
                read_factor, write_factor = OPENAI_CACHE_READ_FACTOR, 1.0
            delta = -self._last_cached_tokens * self.input_cost * (1 - read_factor) + (
                self._last_written_tokens * self.input_cost * (write_factor - 1)
            )
            if (
                delta
                and hasattr(tracking.TRACKER, "instance")
                and isinstance(tracking.TRACKER.instance, tracking.LLMTracker)
            ):
                tracking.TRACKER.instance(0, 0, delta)
        return res


# Fraction of the input rate that OpenAI-shape models (e.g. azure/gpt-5.5) charge for cached
# input tokens; they cache prompt prefixes automatically and carry no cache-write premium.
# OpenAI's cached-input discount is ~0.5x. Claude instead uses ANTHROPIC_CACHE_PRICING_FACTOR.
OPENAI_CACHE_READ_FACTOR = float(os.environ.get("WA_OPENAI_CACHE_READ_FACTOR", "0.5"))

# $/token (input, output) for models served via the proxy, used when WA_INPUT_COST /
# WA_OUTPUT_COST are unset. The aws/claude-sonnet-5 rates are the litellm cost-map values.
_PROXY_RATES = {
    "aws/claude-sonnet-5": (1.52e-6, 7.60e-6),
}


@dataclass
class ProxyModelArgs(BaseModelArgs):
    """litellm->Bedrock Claude via an OpenAI-compatible proxy. Text-mode (no tool-calling).

    Prompt caching is on by default (per-episode sticky routing); set WA_CACHE=0 to disable.
    """

    def make_model(self):
        model_cls = ChatModel if os.environ.get("WA_CACHE") == "0" else CacheAwareChatModel
        if _use_rits():
            # RITS (IBM Research Inference, see rits.py) is OpenAI-compatible but authenticates
            # via a custom ``RITS_API_KEY`` header. The OpenAI SDK still needs a non-empty
            # api_key, and rits.py sets it to the same key, so we mirror that. Each RITS
            # deployment serves ONE model, so the base_url is model-specific (RITS_BASE_URL).
            rits_key = os.environ.get("RITS_API_KEY")
            if not rits_key:
                raise KeyError("WA_RITS=1 but RITS_API_KEY is unset")
            base_url = os.environ.get("RITS_BASE_URL", DEFAULT_RITS_BASE_URL)
            client_args: dict = {
                "base_url": base_url,
                "default_headers": {"RITS_API_KEY": rits_key},
            }
            api_key = rits_key
        else:
            client_args = {"base_url": os.environ["OPENAI_BASE_URL"]}  # ".../v1" or ".../anthropic"
            headers = _custom_headers()
            if headers:  # e.g. ANTHROPIC_CUSTOM_HEADERS="x-context-guru-token: cg_live_xxx"
                client_args["default_headers"] = headers
            # Auth may be carried by a custom header (e.g. contextguru) rather than the OpenAI
            # api_key. The OpenAI SDK still requires a non-empty api_key string, so fall back to
            # a placeholder when OPENAI_API_KEY is unset but custom headers provide auth.
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                if not headers:
                    raise KeyError("OPENAI_API_KEY unset and no ANTHROPIC_CUSTOM_HEADERS to auth with")
                api_key = "header-auth"  # placeholder; real auth is in default_headers

        def _pricing_func():
            """Per-token prices so cost tracking is non-zero on the proxy path.

            Priority: WA_INPUT_COST / WA_OUTPUT_COST env vars -> the _PROXY_RATES table ->
            {} (base ChatModel then sets prices to 0 and warns). Cache-read/-write discounts
            are applied on top by CacheAwareChatModel via ANTHROPIC_CACHE_PRICING_FACTOR.
            """
            env_in, env_out = os.environ.get("WA_INPUT_COST"), os.environ.get("WA_OUTPUT_COST")
            if env_in and env_out:
                return {self.model_name: {"prompt": float(env_in), "completion": float(env_out)}}
            if self.model_name in _PROXY_RATES:
                cin, cout = _PROXY_RATES[self.model_name]
                return {self.model_name: {"prompt": cin, "completion": cout}}
            return {}  # unknown model + no override -> base sets 0 and warns

        model = model_cls(
            model_name=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=_pricing_func,
        )
        _install_call_logger(model)
        return model


# Model id as the OpenAI SDK must send it to the proxy — NO "openai/" prefix (that is a
# litellm client-side routing convention; sending it here yields 403 model-access-denied).
# Override per run with WA_AGENT_MODEL (must be a model id the proxy team is allowed to access,
# e.g. claude-sonnet-4-5-20250929, claude-opus-4-8, aws/claude-sonnet-5).
DEFAULT_AGENT_MODEL = "claude-sonnet-4-5-20250929"

PROXY_MODEL_ARGS = ProxyModelArgs(
    model_name=os.environ.get("WA_AGENT_MODEL", DEFAULT_AGENT_MODEL),
    max_total_tokens=200_000,
    max_input_tokens=180_000,
    # Reasoning models (e.g. moonshotai/Kimi-K2.7-Code) spend output tokens on a hidden
    # `reasoning` field before emitting `content`; too small a budget returns content=None
    # (finish_reason=length). Raise via WA_MAX_NEW_TOKENS for those (e.g. 8000).
    max_new_tokens=int(os.environ.get("WA_MAX_NEW_TOKENS", "2000")),
    # Reasoning models (e.g. azure/gpt-5.5) reject any temperature but the default 1
    # ("Unsupported value: 'temperature' does not support 0.1"). Override with
    # WA_TEMPERATURE=1 for those; default preserved for the Claude models.
    temperature=float(os.environ.get("WA_TEMPERATURE", "0.1")),
    vision_support=False,
)
