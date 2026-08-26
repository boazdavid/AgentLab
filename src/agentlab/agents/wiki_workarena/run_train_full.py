"""Re-run the FULL WorkArena train grid (all 33 L1 task types x seeds 42-49 = 264).

Uniform, single-config reproduction of the stage-2 TRAIN rollouts (control arm),
so the trajectories can be re-converted with the fixed converter (semantic actions
+ full AX tree) and the wiki re-learned from a consistent dataset.

Config mirrors run_train.build_train_study: stock GenericAgent, FLAGS_GPT_4o with
use_error_logs, Sonnet-5 via the litellm->Bedrock proxy, L1 max_steps=15.

Usage:
    python -m agentlab.agents.wiki_workarena.run_train_full --n-jobs 6
"""

import argparse
from copy import deepcopy

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.study import make_study

from agentlab.agents.wiki_workarena.proxy_model import PROXY_MODEL_ARGS
from agentlab.agents.wiki_workarena.splits import build_benchmark, split_pairs

COMMENT = "workarena-wiki-train-full"


def build_full_train_study(n_train=8, n_test=8, seed=42, comment=COMMENT):
    """Assemble the full train-grid study (all task types). No network / no .run()."""
    train_pairs, _ = split_pairs(n_train=n_train, n_test=n_test, seed=seed)
    print(f"TRAIN_PAIRS (full grid): {len(train_pairs)} pairs")
    bench = build_benchmark(train_pairs)
    flags = deepcopy(FLAGS_GPT_4o)
    flags.use_error_logs = True
    agent_args = GenericAgentArgs(chat_model_args=PROXY_MODEL_ARGS, flags=flags)
    return make_study(agent_args=[agent_args], benchmark=bench, comment=comment)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=8)
    args = ap.parse_args()

    study = build_full_train_study(n_train=args.n_train)
    study.run(n_jobs=args.n_jobs)
    print("TRAIN_FULL_STUDY_DIR", study.dir)


if __name__ == "__main__":
    main()
