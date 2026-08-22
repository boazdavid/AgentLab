"""Launch the TEST studies for the WorkArena wiki-knowledge pilot.

Two arms, run on the IDENTICAL test pairs:

* ``control``   — stock GenericAgent, stock benchmark action set.
* ``retrieval`` — the INDEX-BATCH arm (6b): the full wiki catalog (index.md) is
  injected into the agent preamble via ``extra_instructions`` and the agent gets
  ONE extra action, ``get_articles([slugs])`` (no query_articles, no embedding).

Usage:
    python -m agentlab.agents.wiki_workarena.run_test_arms \
        --arm both --wiki-dir /path/to/built_wiki --n-jobs 4
"""

import argparse
import os
from copy import deepcopy

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.study import make_study

from agentlab.agents.wiki_workarena.pilot_common import (
    PILOT_TASKS,
    render_index_preamble,
    select_pairs,
)
from agentlab.agents.wiki_workarena.cached_agent import CachedSystemAgentArgs
from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS
from agentlab.agents.wiki_workarena.retrieval_actions import WikiActionSetArgs
from agentlab.agents.wiki_workarena.splits import build_benchmark


def build_arm(arm, test_pairs, wiki_dir=None, max_steps=15):
    """Return ``(agent_args, benchmark)`` for one arm. Pure construction (no run).

    * ``control``   — stock flags, stock benchmark action set.
    * ``retrieval`` — WikiActionSetArgs exposing ONLY get_articles + the wiki
      index.md catalog injected into the CACHED system prompt (via
      ``CachedSystemAgentArgs.cached_system_suffix``) rather than the volatile
      human message. The catalog text is identical to before — only its location
      moved, so prompt caching reads it on later steps instead of re-sending it.

    The caller is responsible for calling ``make_study`` (and, for retrieval,
    setting ``KNOWLEDGE_URL`` in the environment).
    """
    bench = build_benchmark(test_pairs, max_steps=max_steps)
    flags = deepcopy(FLAGS_GPT_4o)
    flags.use_error_logs = True

    if arm == "retrieval":
        if wiki_dir is None:
            raise ValueError("retrieval arm requires --wiki-dir (for index.md)")
        bench.high_level_action_set_args = WikiActionSetArgs(
            subsets=("workarena", "custom"),
            action_names=("get_articles",),
            multiaction=False,
        )
        flags.use_past_error_logs = True
        # The catalog goes into the cached system prompt, NOT extra_instructions
        # (which AgentLab renders in the volatile, uncached human message).
        agent_args = CachedSystemAgentArgs(
            chat_model_args=PROXY_MODEL_ARGS,
            flags=flags,
            cached_system_suffix=render_index_preamble(wiki_dir),
        )
    elif arm == "control":
        agent_args = GenericAgentArgs(chat_model_args=PROXY_MODEL_ARGS, flags=flags)
    else:
        raise ValueError(f"unknown arm: {arm}")

    return agent_args, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "retrieval", "both"], default="both")
    ap.add_argument("--tasks", default=",".join(PILOT_TASKS),
                    help="comma-separated task-id substrings (default: 6 pilot types)")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap test pairs for shakeout (0 = all)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=15, help="per-episode step cap")
    ap.add_argument("--wiki-dir", default=None, help="built wiki dir (for index.md)")
    ap.add_argument("--knowledge-url",
                    default=os.environ.get("KNOWLEDGE_URL", "http://127.0.0.1:8799"))
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    _train_pairs, test_pairs = select_pairs(tasks, n_train=args.n_train, n_test=args.n_test)
    total = len(test_pairs)
    if args.limit and args.limit > 0:
        test_pairs = test_pairs[:args.limit]
        print(f"TEST_PAIRS capped: using {len(test_pairs)} of {total} (--limit {args.limit})")
    else:
        print(f"TEST_PAIRS: using all {total}")

    arms = ["control", "retrieval"] if args.arm == "both" else [args.arm]
    if "retrieval" in arms:
        # The exec'd get_articles action reads KNOWLEDGE_URL from the environment.
        os.environ["KNOWLEDGE_URL"] = args.knowledge_url
        print(f"KNOWLEDGE_URL {args.knowledge_url}")

    for arm in arms:
        agent_args, bench = build_arm(arm, test_pairs, wiki_dir=args.wiki_dir, max_steps=args.max_steps)
        study = make_study(
            agent_args=[agent_args], benchmark=bench, comment=f"workarena-wiki-{arm}"
        )
        study.run(n_jobs=args.n_jobs)
        print(f"{arm.upper()}_STUDY_DIR", study.dir)


if __name__ == "__main__":
    main()
