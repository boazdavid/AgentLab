"""GenericAgent variant that relocates large STABLE content into the CACHED system prompt.

Motivation
----------
``CacheAwareChatModel`` (see ``proxy_model.py``) marks a prompt-cache breakpoint on the
system message — the one block that is byte-identical across every step of an episode.
AgentLab's stock system prompt is tiny, so caching it saves almost nothing. The wiki
retrieval arm's real bulk (the full wiki catalog / index preamble) is otherwise injected
via ``flags.extra_instructions``, which AgentLab renders into the *human* message — volatile
content that is re-sent uncached on every one of the ~15 steps.

``CachedSystemAgent`` appends that catalog to the system prompt instead, so it is
cache-written once per episode and cache-read on every later step. The agent is otherwise
identical to ``GenericAgent``.

Design
------
The only behavioural change is the text of the system message that ``GenericAgent.get_action``
builds inline as ``SystemMessage(dp.SystemPrompt().prompt)`` (generic_agent.py get_action).
Rather than duplicate the whole ~60-line ``get_action`` (fragile against upstream changes),
we override the single documented seam: during ``get_action`` we temporarily point
``dp.SystemPrompt._prompt`` at the augmented text (base + suffix), then delegate to
``super().get_action``. Everything else — truncation, retry, stats, cost tracking — is
inherited unchanged. ``build_system_prompt()`` exposes the exact ``SystemMessage`` that will
be emitted, for offline testing.
"""

from dataclasses import dataclass

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import SystemMessage


class CachedSystemAgent(GenericAgent):
    """A ``GenericAgent`` that appends ``cached_system_suffix`` to the system prompt.

    The suffix rides in the system message (stable per episode), so the
    ``CacheAwareChatModel`` breakpoint caches it once and reads it on later steps.
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        cached_system_suffix: str = "",
    ):
        self.cached_system_suffix = cached_system_suffix
        super().__init__(chat_model_args=chat_model_args, flags=flags, max_retry=max_retry)

    def _augmented_system_prompt_text(self) -> str:
        """Base ``SystemPrompt`` text with the cached suffix appended (if any)."""
        base = dp.SystemPrompt().prompt
        if self.cached_system_suffix:
            return base + "\n\n" + self.cached_system_suffix
        return base

    def build_system_prompt(self) -> SystemMessage:
        """The exact ``SystemMessage`` ``get_action`` will emit (used by the cache)."""
        return SystemMessage(self._augmented_system_prompt_text())

    def get_action(self, obs):
        # Temporarily augment the single line generic_agent.get_action reads:
        #   system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        # then delegate so all other get_action logic is inherited unchanged.
        original_prompt = dp.SystemPrompt._prompt
        try:
            dp.SystemPrompt._prompt = self._augmented_system_prompt_text()
            return super().get_action(obs)
        finally:
            dp.SystemPrompt._prompt = original_prompt


@dataclass
class CachedSystemAgentArgs(GenericAgentArgs):
    """``GenericAgentArgs`` carrying a ``cached_system_suffix`` for the system prompt.

    Inherits ``set_benchmark`` (which deepcopies the benchmark's action set — preserving
    the ``WikiActionSetArgs`` wiring) and ``__post_init__`` unchanged.
    """

    cached_system_suffix: str = ""

    def make_agent(self):
        # Replicate GenericAgentArgs.make_agent's construction, carrying the suffix.
        return CachedSystemAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            cached_system_suffix=self.cached_system_suffix,
        )
