"""Smoke test for the WorkArena wiki-knowledge setup.

Two modes:

* ``--offline`` (default here when creds are absent): assemble the whole agent +
  benchmark + action-set stack WITHOUT connecting to ServiceNow. Verifies imports,
  the proxy ModelArgs, the seed-split benchmark builder, and (with ``--retrieval``)
  the wiki custom-action wiring all construct cleanly. Reports which live creds are
  still missing. This is the "does the plumbing assemble" gate.

* ``--live``: actually run ONE WorkArena-L1 episode end-to-end (browser vs. the live
  ServiceNow instance) with a stock GenericAgent driven by the litellm->Bedrock proxy.
  Requires SNOW_INSTANCE_URL/UNAME/PWD, OPENAI_BASE_URL/OPENAI_API_KEY, HF auth, and
  AGENTLAB_EXP_ROOT (populate ~/Documents/GitHub/AgentLab/.env). Writes a
  summary_info.json under the study dir; success is not required here — reachability is.

Usage:
    python -m agentlab.agents.wiki_workarena.smoke_one_task            # offline assembly
    python -m agentlab.agents.wiki_workarena.smoke_one_task --live     # full live run
    python -m agentlab.agents.wiki_workarena.smoke_one_task --live --retrieval  # + wiki arm
"""

import argparse
import os
from copy import deepcopy

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.study import make_study

from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS, _use_rits
from agentlab.agents.wiki_workarena.splits import build_benchmark, split_pairs
from agentlab.agents.wiki_workarena.retrieval_actions import WikiActionSetArgs

# Genuinely required for a live run. HF_TOKEN gates the gated WorkArena-Instances
# pool, from which an instance is auto-allocated when SNOW_INSTANCE_* are unset.
# Model auth is either OPENAI_API_KEY or a custom header (ANTHROPIC_CUSTOM_HEADERS,
# e.g. the contextguru token) — see _missing_live_env(). In RITS mode the endpoint
# and auth come from RITS_API_KEY/RITS_BASE_URL instead of the OPENAI_* vars.
LIVE_ENV = [
    "AGENTLAB_EXP_ROOT",
    "HF_TOKEN",
]


def _missing_live_env():
    """Required live env vars that are unset. Model endpoint + auth are satisfied by
    EITHER the RITS vars (RITS_API_KEY, in RITS mode) OR OPENAI_BASE_URL plus one of
    OPENAI_API_KEY / ANTHROPIC_CUSTOM_HEADERS (proxy header-based auth)."""
    missing = [v for v in LIVE_ENV if not os.getenv(v)]
    if _use_rits():
        if not os.getenv("RITS_API_KEY"):
            missing.append("RITS_API_KEY")
    else:
        if not os.getenv("OPENAI_BASE_URL"):
            missing.append("OPENAI_BASE_URL")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_CUSTOM_HEADERS"):
            missing.append("OPENAI_API_KEY|ANTHROPIC_CUSTOM_HEADERS")
    return missing
# Optional: pin a specific ServiceNow instance instead of the managed pool. If any
# is set, all three should be (Path B, bring-your-own-instance).
SNOW_OPTIONAL_ENV = ["SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD"]


def _build_study(retrieval: bool):
    """Assemble a 1-episode study. Pure construction — no network."""
    _, test = split_pairs(n_train=1, n_test=1)
    bench = build_benchmark(test[:1])  # exactly one (task, seed)
    if retrieval:
        bench.high_level_action_set_args = WikiActionSetArgs(
            subsets=("workarena", "custom"), multiaction=False
        )
    flags = deepcopy(FLAGS_GPT_4o)
    flags.use_error_logs = True
    flags.use_past_error_logs = True
    agent_args = GenericAgentArgs(chat_model_args=PROXY_MODEL_ARGS, flags=flags)
    study = make_study(agent_args=[agent_args], benchmark=bench, comment="workarena-wiki-smoke")
    return study, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run one real episode vs ServiceNow")
    ap.add_argument("--retrieval", action="store_true", help="use the wiki retrieval action set")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    missing = _missing_live_env()

    study, bench = _build_study(args.retrieval)
    pair = [(e.task_name, e.task_seed) for e in bench.env_args_list]
    aset = bench.high_level_action_set_args.make_action_set()
    # python_includes is a single source string; use plain containment (not per-item).
    has_custom = "def query_memories" in aset.python_includes and "def get_memories" in aset.python_includes

    print("=== WorkArena smoke: assembly ===")
    print(f"  episode           : {pair}")
    print(f"  model             : {PROXY_MODEL_ARGS.model_name} (text mode)")
    print(f"  arm               : {'retrieval (wiki actions)' if args.retrieval else 'control (stock)'}")
    print(f"  custom actions in set: {has_custom}")
    print(f"  study dir         : {study.dir}")
    print("  ASSEMBLY OK — agent, benchmark, and action set all construct.")

    if not args.live:
        print("\n=== offline mode (no ServiceNow connection attempted) ===")
        if missing:
            print(f"  live run needs these still-missing env vars: {missing}")
            print("  populate ~/Documents/GitHub/AgentLab/.env, then rerun with --live")
        else:
            print("  all required live env vars present — you can rerun with --live")
        snow_set = [v for v in SNOW_OPTIONAL_ENV if os.getenv(v)]
        if snow_set:
            print(f"  SNOW_INSTANCE_* set ({snow_set}) — pinning your own instance (Path B)")
        else:
            print("  SNOW_INSTANCE_* unset — an instance will be auto-allocated from the "
                  "gated WorkArena-Instances pool (Path A)")
        return

    if missing:
        raise SystemExit(f"--live requires env vars, missing: {missing}")

    print("\n=== live run: one WorkArena-L1 episode vs ServiceNow ===")
    study.run(n_jobs=args.n_jobs)
    print("LIVE_SMOKE_STUDY_DIR", study.dir)


if __name__ == "__main__":
    main()
