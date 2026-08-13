"""Custom AgentLab ``ModelArgs`` that routes an agent through an
OpenAI-compatible litellm -> Bedrock proxy in text mode (no tool-calling).

Credentials are read from the environment (``OPENAI_BASE_URL`` +
``OPENAI_API_KEY``), optionally populated from a local ``.env``. Nothing here
prints or hardcodes secrets.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel

# SNOW_*, OPENAI_*, AGENTLAB_EXP_ROOT, HF token. Missing file is fine.
load_dotenv(Path.home() / "Documents/GitHub/AgentLab/.env")


@dataclass
class ProxyModelArgs(BaseModelArgs):
    """litellm->Bedrock Claude via an OpenAI-compatible proxy. Text-mode (no tool-calling)."""

    def make_model(self):
        return ChatModel(
            model_name=self.model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            client_class=OpenAI,
            client_args={"base_url": os.environ["OPENAI_BASE_URL"]},  # ".../v1"
        )


PROXY_MODEL_ARGS = ProxyModelArgs(
    model_name="openai/claude-sonnet-4-5-20250929",
    max_total_tokens=200_000,
    max_input_tokens=180_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=False,
)
