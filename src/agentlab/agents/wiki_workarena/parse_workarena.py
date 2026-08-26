"""Convert AgentLab / WorkArena episodes into try_wikis ``Trajectory`` JSON.

This lives inside the AgentLab package because loading an episode requires
unpickling browsergym StepInfo objects (``browsergym``/``agentlab`` must be
importable). The try_wikis package is NOT installed here, so we emit plain
dicts matching its ``Trajectory`` *input* schema — no computed fields:

    {"file_path": str, "system_prompt": "", "tools": [],
     "messages": [
        {"role": "user"|"assistant", "content": str,
         "tool_calls": [{"tool_name": str, "args": {...},
                         "success": bool, "error_text": str|None,
                         "response": str|None}]}
     ]}

Design — two layers so the mapping is unit-testable offline (no pickles):

1. ``episode_to_trajectory_dict(goal, steps, file_path)`` — the PURE mapping.
   ``steps`` is a list of plain dicts (see ``load_episode`` for the fields).
   No browsergym dependency; this is what the offline tests exercise.
2. ``load_episode(exp_dir)`` — the I/O wrapper that reads the pickled episode
   via ``browsergym.experiments.loop.get_exp_result`` and produces those dicts.

Plus ``iter_study`` (enumerate + filter episodes) and a ``__main__`` CLI.
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path

# Markers that the retrieval custom actions (retrieval_actions.py) prepend to
# the text they deliver via the NEXT step's last_action_error. This is the
# intended delivery channel for wiki text, NOT a failure.
_WIKI_MARKERS = ("WIKI SEARCH RESULTS:", "WIKI ARTICLES:")
_RETRIEVAL_ACTIONS = {"query_articles", "get_articles"}

# Positional parameter names for the common WorkArena/browsergym high-level
# actions, in signature order. Used to turn positional action args into named
# args without importing browsergym (keeps the pure layer offline). Optional
# trailing params (button/modifiers/...) are included so extra positionals
# still map. Unknown actions fall back to {"args": [...]}.
_ACTION_PARAMS = {
    "click": ["bid", "button", "modifiers"],
    "dblclick": ["bid", "button", "modifiers"],
    "fill": ["bid", "value", "enable_autocomplete_menu"],
    "hover": ["bid"],
    "focus": ["bid"],
    "check": ["bid"],
    "uncheck": ["bid"],
    "clear": ["bid"],
    "select_option": ["bid", "options"],
    "press": ["bid", "key_comb"],
    "drag_and_drop": ["from_bid", "to_bid"],
    "upload_file": ["bid", "file"],
    "goto": ["url"],
    "go_back": [],
    "go_forward": [],
    "scroll": ["delta_x", "delta_y"],
    "keyboard_press": ["key"],
    "keyboard_type": ["text"],
    "keyboard_insert_text": ["text"],
    "tab_focus": ["index"],
    "tab_close": [],
    "new_tab": [],
    "noop": ["wait_ms"],
    "send_msg_to_user": ["text"],
    "report_infeasible": ["reason"],
    # wiki retrieval custom actions
    "query_articles": ["q"],
    "get_articles": ["slugs"],
}


def resolve_bid_target(axtree_txt, bid):
    """Resolve a browsergym ``bid`` to its element ``{"role", "name"}`` from an AX tree.

    Accessibility-tree lines look like ``[a484] textbox 'Short description', required``.
    Returns ``None`` when the bid is absent (e.g. resolved against a truncated tree).

    This resolution MUST run against the FULL ``axtree_txt`` at capture time: acted
    elements (form fields, buttons) usually sit deep in the tree, below the nav/header
    that fills a truncated prefix, and browsergym namespaces sub-frame bids (``a484``).
    Downstream consumers only see the acted element's semantics if we resolve here.
    """
    if not axtree_txt or bid is None:
        return None
    m = re.search(r"\[" + re.escape(str(bid)) + r"\]\s+([A-Za-z]+)(?:\s+'([^']*)')?", axtree_txt)
    if not m:
        return None
    return {"role": m.group(1), "name": m.group(2) or ""}


def _url_path(url):
    """Path component of a URL: strip scheme+host and the volatile ``?query``.

    Path-only keeps the action identity stable across the rotating ServiceNow hosts
    (workarenapublic19/20/22/23) — the host is per-seed noise, the path is the page.
    """
    if not url:
        return ""
    u = url.split("?", 1)[0]
    m = re.match(r"https?://[^/]+(/.*)$", u)
    if m:
        return m.group(1)
    return u if u.startswith("/") else ""


def _semantic_tool_name(action_name, target, url):
    """Bake a bid-based browser action into a semantic identity: ``name[role:name@/path]``.

    Owns the browser-specific translation at save time: the acted element (resolved
    from the FULL AX tree) and the page path become a stable action key, so the
    downstream try_wikis loader stays a dumb JSON reader and the gate stays generic.
    Actions without a resolved target (noop, unresolved bid) keep the base name.
    """
    if not target:
        return action_name
    label = f"{target.get('role') or '?'}:{target.get('name') or ''}"
    path = _url_path(url)
    if path:
        label += f"@{path}"
    return f"{action_name}[{label}]"


def _param_names(action_name):
    """Return the positional parameter names for ``action_name``.

    Prefer the static table; if absent, try ``inspect.signature`` on the real
    browsergym action function (skipping playwright-injected ``page``-like
    params). Returns ``None`` when nothing is derivable, so the caller falls
    back to positional ``{"args": [...]}``.
    """
    if action_name in _ACTION_PARAMS:
        return _ACTION_PARAMS[action_name]
    try:  # best-effort; unavailable/offline -> positional fallback
        import inspect

        from browsergym.core.action import functions as F

        fn = getattr(F, action_name, None)
        if fn is None:
            return None
        names = []
        for p in inspect.signature(fn).parameters.values():
            ann = p.annotation
            # skip the playwright page handle that some primitives take first
            if p.name == "page" or "playwright" in str(ann):
                continue
            names.append(p.name)
        return names
    except Exception:
        return None


def _named_args(action_name, action_args):
    """Map an action's args to a NAMED dict when derivable, else positional.

    ``action_args`` may be a list of positional values, a dict wrapping them as
    ``{"args": [...]}``, or an already-named dict. Never raises.
    """
    # normalise to a positional list where possible
    if isinstance(action_args, list):
        positional = list(action_args)
    elif isinstance(action_args, dict):
        if set(action_args.keys()) == {"args"} and isinstance(action_args["args"], list):
            positional = list(action_args["args"])
        elif not action_args:
            positional = []
        else:
            # already named (or partially) — pass through untouched
            return dict(action_args)
    else:
        positional = []

    names = _param_names(action_name)
    if names is None:
        return {"args": positional}
    return {names[i]: v for i, v in enumerate(positional) if i < len(names)}


def _reasoning_from_think(think):
    """Extract reasoning text that precedes any ``<action>`` tag.

    Raw ``think`` often looks like ``"...text...\n\n<action>\nclick('79')\n</action>"``.
    We keep only the text before ``<action>`` (the action itself is captured in
    the tool_call). Empty -> ``"(no reasoning text)"``.
    """
    text = think or ""
    idx = text.find("<action>")
    if idx != -1:
        text = text[:idx]
    text = text.strip()
    return text if text else "(no reasoning text)"


def _strip_wiki_marker(text):
    """Drop the leading ``WIKI ...:`` marker line from delivered wiki text."""
    for marker in _WIKI_MARKERS:
        if text.startswith(marker):
            rest = text[len(marker):]
            return rest.lstrip("\n")
    return text


def _outcome(action_name, next_step):
    """Compute (success, error_text, response) for an action from the NEXT step.

    The success/error signal for the action at step ``i`` is carried by step
    ``i+1``'s ``last_action_error``. ``next_step`` is that following step dict
    (or ``None`` when the acting step is the last one).

    - No next step -> ``(True, None, None)``.
    - Retrieval action (query_articles/get_articles) whose ``next_step`` error
      starts with a WIKI marker: wiki text was delivered -> success, response =
      the (marker-stripped) text. Response STILL carries the wiki text for
      retrieval — that genuinely is the tool's result the agent reads.
    - A non-empty ``next_step`` error that is not a WIKI marker is a real
      failure -> ``(False, error, None)``.
    - Otherwise a successful regular browser action -> ``(True, None, None)``.
      Its observation is NOT put in the response; the page the agent saw lives
      in the ``[observation]`` user message that PRECEDES the action.
    """
    if next_step is None:
        return True, None, None

    next_error = next_step.get("last_action_error")
    nxt = (next_error or "").strip() if next_error is not None else ""
    is_wiki = any(nxt.startswith(m) for m in _WIKI_MARKERS)

    if action_name in _RETRIEVAL_ACTIONS and is_wiki:
        return True, None, _strip_wiki_marker(nxt)
    if nxt and not is_wiki:
        return False, nxt, None

    return True, None, None


def episode_to_trajectory_dict(goal, steps, file_path):
    """Pure mapping: (goal, list[step dict], file_path) -> Trajectory dict.

    See module docstring for the emitted schema. Has no browsergym dependency.
    """
    messages = [
        {"role": "user", "content": f"TASK GOAL: {goal}", "tool_calls": []}
    ]

    for i, step in enumerate(steps):
        action_name = step.get("action_name")
        next_step = steps[i + 1] if i + 1 < len(steps) else None

        # observe -> act: emit the step's OWN observation (the page the agent
        # saw to DECIDE this step's action) as a user message BEFORE the action.
        url = step.get("url") or ""
        axtree_head = step.get("axtree_head") or ""
        messages.append(
            {
                "role": "user",
                "content": f"[observation] url={url}\n{axtree_head}",
                "tool_calls": [],
            }
        )

        if action_name:
            success, error_text, response = _outcome(action_name, next_step)
            args = _named_args(action_name, step.get("action_args"))
            # Bake the acted element (role/name, resolved from the FULL AX tree) and the
            # page path into a semantic action identity, and DROP the ephemeral ``bid``
            # (a per-render DOM handle — noise for process identity). The downstream
            # try_wikis loader is then a dumb reader and the gate stays domain-agnostic.
            target = step.get("action_target")
            tool_name = _semantic_tool_name(action_name, target, url)
            if isinstance(args, dict):
                args = {k: v for k, v in args.items() if k != "bid"}
            tool_call = {
                "tool_name": tool_name,
                "args": args,
                "success": success,
                "error_text": error_text,
                "response": response,
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": _reasoning_from_think(step.get("think")),
                    "tool_calls": [tool_call],
                }
            )
        # Terminal step (no action) contributes only its observation (appended above).
        # reward/terminated are deliberately EXCLUDED: they are benchmark ground-truth
        # signals absent from real production agent logs, so they must not leak into the
        # Trajectory (this also prevents the wiki builder from keying on the outcome).

    return {
        "file_path": file_path,
        "messages": messages,  # system_prompt/tools omitted -> Trajectory defaults ("", [])
    }


# --- I/O layer (requires browsergym; not exercised by offline tests) --------

def _goal_from_obs(obs):
    """Extract goal text from ``obs['goal_object']`` text parts (or ``goal``)."""
    if not obs:
        return ""
    goal_object = obs.get("goal_object")
    if goal_object:
        parts = []
        for item in goal_object:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        if parts:
            return "\n".join(parts).strip()
    return (obs.get("goal") or "").strip()


def _parse_action(action_str):
    """Parse an action string like ``click('79')`` -> (name, positional args).

    Uses ``ast`` to safely evaluate the call and its literal arguments. Returns
    ``(None, [])`` for empty/unparseable actions. Keyword args are folded in as
    a trailing named dict only if present; positional args returned as a list.
    """
    if not action_str or not action_str.strip():
        return None, []
    try:
        expr = ast.parse(action_str.strip(), mode="eval").body
        if not isinstance(expr, ast.Call):
            return None, []
        name = expr.func.id if isinstance(expr.func, ast.Name) else None
        args = [ast.literal_eval(a) for a in expr.args]
        # keyword args (rare) -> merge as named dict passed through _named_args
        if expr.keywords:
            named = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg}
            # return a dict form; _named_args passes named dicts through, but we
            # also want the positionals named, so resolve here.
            resolved = _named_args(name, args)
            if isinstance(resolved, dict) and "args" not in resolved:
                resolved.update(named)
                return name, resolved
            return name, {"args": args, **named}
        return name, args
    except Exception:
        return None, []


def load_episode(exp_dir, axtree_chars=None):
    """Load one episode from ``exp_dir`` into (goal, list[step dict]).

    Reads the pickled steps via browsergym's ``get_exp_result``. Each emitted
    step dict has the fields consumed by ``episode_to_trajectory_dict`` plus the
    raw ``action`` string and its parsed name/args.

    ``axtree_chars=None`` (default) keeps the FULL accessibility tree in each
    observation so the downstream ``extract`` stage sees the whole page, not just
    a truncated prefix. Pass an int only to cap it (e.g. for cheap smoke runs).
    The acted-element resolution always uses the full tree regardless.
    """
    from browsergym.experiments.loop import get_exp_result

    result = get_exp_result(str(exp_dir))
    steps_info = result.steps_info

    goal = ""
    for si in steps_info:
        goal = _goal_from_obs(getattr(si, "obs", None))
        if goal:
            break

    steps = []
    for si in steps_info:
        obs = getattr(si, "obs", None) or {}
        agent_info = getattr(si, "agent_info", None) or {}
        action_str = getattr(si, "action", None)
        action_name, action_args = _parse_action(action_str)
        axtree = obs.get("axtree_txt") or ""
        # Resolve the acted element against the FULL axtree (not the truncated head):
        # the target usually sits deep in the tree, so head-only resolution misses it.
        named = _named_args(action_name, action_args) if action_name else {}
        bid = named.get("bid") or named.get("from_bid") if isinstance(named, dict) else None
        action_target = resolve_bid_target(axtree, bid) if bid is not None else None
        steps.append(
            {
                "think": agent_info.get("think") or "",
                "action": action_str,
                "action_name": action_name,
                "action_args": action_args,
                "action_target": action_target,
                "url": obs.get("url") or "",
                "axtree_head": axtree if axtree_chars is None else axtree[:axtree_chars],
                "last_action_error": obs.get("last_action_error") or None,
                "reward": getattr(si, "reward", 0),
                "terminated": getattr(si, "terminated", None),
            }
        )
    return goal, steps


def iter_study(study_dir, success_only=True):
    """Yield episode dirs under ``study_dir``.

    An episode dir is any subdirectory containing ``summary_info.json``. When
    ``success_only`` (default), keep only episodes with ``cum_reward == 1.0`` —
    the wiki learns from successes, mirroring the other trajectory parsers.
    """
    study_dir = Path(study_dir)
    for summary in sorted(study_dir.rglob("summary_info.json")):
        ep_dir = summary.parent
        if not success_only:
            yield ep_dir
            continue
        try:
            info = json.loads(summary.read_text())
        except Exception:
            continue
        if float(info.get("cum_reward", 0) or 0) == 1.0:
            yield ep_dir


def _episode_out_name(exp_dir):
    """Build ``<task>_<seed>.json`` from an episode dir name.

    Dir names look like ``..._GenericAgent-<model>_on_<task>_<seed>``. We keep
    the task + seed tail; fall back to the sanitized dir name if that fails.
    """
    base = Path(exp_dir).name
    if "_on_" in base:
        tail = base.split("_on_", 1)[1]  # e.g. workarena.servicenow.all-menu_43
        return tail.replace("/", "_") + ".json"
    return base.replace("/", "_") + ".json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", required=True, help="AgentLab study dir containing episode subdirs")
    ap.add_argument("--out-dir", required=True, help="where to write per-episode Trajectory JSON")
    ap.add_argument("--all", action="store_true", help="include failed episodes (default: successes only)")
    ap.add_argument("--axtree-chars", type=int, default=None,
                    help="cap axtree_txt chars per step (default: None = full tree)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for ep_dir in iter_study(args.study_dir, success_only=not args.all):
        goal, steps = load_episode(ep_dir, axtree_chars=args.axtree_chars)
        traj = episode_to_trajectory_dict(goal, steps, file_path=str(ep_dir))
        out_path = out_dir / _episode_out_name(ep_dir)
        out_path.write_text(json.dumps(traj, indent=2))
        n += 1
        print(f"wrote {out_path}")
    print(f"done: {n} episode(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
