"""Wiki-knowledge retrieval custom actions for BrowserGym/WorkArena.

Two extra high-level actions are exposed to the agent:

- ``query_articles(q)`` — natural-language search over the wiki index.
- ``get_articles(slugs)`` — batch-fetch full article bodies for a list of slugs.

Both call the Task-7 HTTP shim (default ``http://127.0.0.1:8799``, override with
the ``KNOWLEDGE_URL`` env var):

    POST {KNOWLEDGE_URL}/query_articles  {"q": str}          -> {"text": str}
    POST {KNOWLEDGE_URL}/get_articles     {"slugs": [str]}    -> {"text": str}

IMPORTANT runtime constraints (why the code looks the way it does):

* BrowserGym executes a custom action by copying the function's *source* via
  ``inspect.getsource`` into a preamble string and running it with a bare
  ``exec(code, {"page": ..., "send_message_to_user": ..., "DEMO_MODE": ...})``.
  That namespace contains only those few injected names plus Python builtins —
  it does NOT contain this module's imports or any class defined here. Therefore:
    - every import (``os``, ``json``, ``urllib.request``) lives *inside* the
      function body, and
    - the action raises a built-in ``Exception`` (never a custom class), because
      only builtins are guaranteed to resolve inside that exec namespace.
* The raised message is caught by BrowserGym and surfaced to the agent on the
  next step as ``last_action_error`` (rendered under "## Error from previous
  action:"). This is the intended channel for returning retrieved text — it is
  normal, not a failure.
* Action arguments are ``repr()``-serialized then re-parsed, so they must be
  simple types (``str``, ``list[str]``).
"""

from dataclasses import dataclass

from browsergym.experiments.benchmark.base import HighLevelActionSetArgs


def query_articles(q: str):
    """Search the wiki knowledge base for articles relevant to what you are trying to do.

    Describe the operation in natural language. Returns a short ranked list of
    matching articles (title + description + link target). Pick the relevant link
    targets, then call get_articles with them. The results are delivered on the
    NEXT step under "## Error from previous action:" (this is normal, not an error).

    Examples:
        query_articles("how to create an incident")
    """
    import json
    import os
    import urllib.request

    url = os.environ.get("KNOWLEDGE_URL", "http://127.0.0.1:8799")
    req = urllib.request.Request(
        url + "/query_articles",
        data=json.dumps({"q": q}).encode(),
        headers={"Content-Type": "application/json"},
    )
    text = json.load(urllib.request.urlopen(req, timeout=30))["text"]
    raise Exception("WIKI SEARCH RESULTS:\n" + text)


def get_articles(slugs: list):
    """Fetch the full bodies of several wiki articles at once (batch).

    Pass a list of link targets copied verbatim from query_articles output. The
    article bodies are delivered on the NEXT step under "## Error from previous
    action:" (this is normal, not an error).

    Examples:
        get_articles(["concepts/create-incident.md", "concepts/assign-to-group.md"])
    """
    import json
    import os
    import urllib.request

    url = os.environ.get("KNOWLEDGE_URL", "http://127.0.0.1:8799")
    req = urllib.request.Request(
        url + "/get_articles",
        data=json.dumps({"slugs": slugs}).encode(),
        headers={"Content-Type": "application/json"},
    )
    text = json.load(urllib.request.urlopen(req, timeout=30))["text"]
    raise Exception("WIKI ARTICLES:\n" + text)


# Name -> function registry. WikiActionSetArgs stores a tuple of NAMES (not raw
# callables) so it stays deepcopy/pickle-serializable — AgentLab deepcopies the
# action-set args in set_benchmark.
_ACTIONS = {"query_articles": query_articles, "get_articles": get_articles}


@dataclass
class WikiActionSetArgs(HighLevelActionSetArgs):
    """HighLevelActionSetArgs that wires in a SUBSET of the wiki retrieval actions.

    Use ``"custom"`` as one of the ``subsets`` (e.g. ``("workarena", "custom")``)
    so the base HighLevelActionSet includes the custom_actions.

    ``action_names`` selects which of the module-level actions to expose. It is a
    tuple of NAMES (not callables) so the dataclass stays serializable through the
    deepcopy AgentLab performs in ``set_benchmark``. The default exposes both
    (backward compatible). Index-batch uses ``action_names=("get_articles",)``.
    """

    action_names: tuple = ("query_articles", "get_articles")

    def make_action_set(self):
        from browsergym.core.action.highlevel import HighLevelActionSet

        custom_actions = [_ACTIONS[name] for name in self.action_names]
        return HighLevelActionSet(
            subsets=self.subsets,
            custom_actions=custom_actions,
            multiaction=self.multiaction,
            strict=self.strict,
            retry_with_force=self.retry_with_force,
            demo_mode=self.demo_mode,
        )
