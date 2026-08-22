"""Launch the TRAIN study for the WorkArena wiki-knowledge pilot.

Runs a stock GenericAgent (Sonnet-4.5 via the litellm->Bedrock proxy) over the
TRAIN split of the pilot task types. Trajectories from this study feed the wiki
build; no retrieval action set is used here.

Usage:
    python -m agentlab.agents.wiki_workarena.run_train \
        --tasks create-incident,create-change --n-train 8 --limit 2 --n-jobs 4
"""

import argparse
from copy import deepcopy

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.study import make_study

from agentlab.agents.wiki_workarena.pilot_common import PILOT_TASKS, select_pairs
from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS
from agentlab.agents.wiki_workarena.splits import build_benchmark


def build_train_study(tasks, n_train, n_test, limit, comment="workarena-wiki-train"):
    """Assemble the train study. Pure construction — no network / no .run()."""
    train_pairs, _test_pairs = select_pairs(tasks, n_train=n_train, n_test=n_test)
    total = len(train_pairs)
    if limit and limit > 0:
        train_pairs = train_pairs[:limit]
        print(f"TRAIN_PAIRS capped: using {len(train_pairs)} of {total} (--limit {limit})")
    else:
        print(f"TRAIN_PAIRS: using all {total}")

    bench = build_benchmark(train_pairs)
    flags = deepcopy(FLAGS_GPT_4o)
    flags.use_error_logs = True
    agent_args = GenericAgentArgs(chat_model_args=PROXY_MODEL_ARGS, flags=flags)
    study = make_study(agent_args=[agent_args], benchmark=bench, comment=comment)
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(PILOT_TASKS),
                    help="comma-separated task-id substrings (default: 6 pilot types)")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train pairs for shakeout (0 = all)")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    study = build_train_study(tasks, args.n_train, args.n_test, args.limit)
    study.run(n_jobs=args.n_jobs)
    print("TRAIN_STUDY_DIR", study.dir)


if __name__ == "__main__":
    main()
