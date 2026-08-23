"""Unit tests for WikiAwareError — the render-layer relabeling of wiki payloads.

The retrieval actions can only return text by raising an Exception, which
BrowserGym surfaces as ``last_action_error`` and dynamic_prompting's ``Error``
element renders under "Error from previous action:". WikiAwareError relabels the
marked wiki payloads to a result heading while leaving genuine errors untouched.
"""

from agentlab.agents.wiki_workarena.cached_agent import WikiAwareError


def _render(text, visible=True, prefix="## "):
    return WikiAwareError(text, visible=visible, prefix=prefix).prompt


def test_articles_marker_gets_result_heading_and_strips_marker():
    out = _render("WIKI ARTICLES:\nfull body of the article")
    assert "## Retrieved wiki articles:" in out
    assert "full body of the article" in out
    # the raw marker and the misleading error heading are both gone
    assert "WIKI ARTICLES:" not in out
    assert "Error from previous action:" not in out


def test_search_marker_gets_search_heading():
    out = _render("WIKI SEARCH RESULTS:\nrow1\nrow2")
    assert "## Wiki search results:" in out
    assert "row1" in out
    assert "WIKI SEARCH RESULTS:" not in out
    assert "Error from previous action:" not in out


def test_real_error_is_unchanged():
    out = _render("TimeoutError: locator '242' not found")
    assert "Error from previous action:" in out
    assert "TimeoutError: locator '242' not found" in out


def test_not_visible_renders_empty():
    # PromptElement.prompt returns "" when not visible, regardless of content.
    assert _render("WIKI ARTICLES:\nbody", visible=False) == ""


def test_empty_error_is_unchanged():
    # no marker -> delegate to dp.Error (renders the error heading with empty body)
    out = _render("")
    assert "Error from previous action:" in out
