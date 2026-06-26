from __future__ import annotations

import json
import os
import urllib.request

from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)


# =====================================================
# DISTILBERT INCIDENT CLASSIFIER
# =====================================================

_classifier = None

LABEL_MAP = {
    "LABEL_0": "normal",
    "LABEL_1": "crash",
    "LABEL_2": "intrusion",
    "LABEL_3": "overload",
}


def load_classifier():

    global _classifier

    if _classifier is None:

        _classifier = pipeline(
            "text-classification",
            model="agentcloud/models/cloudsec-model",
            tokenizer="agentcloud/models/cloudsec-model",
        )

    return _classifier


def predict_incident(text: str):

    clf = load_classifier()

    result = clf(text)[0]

    return {
        "label": LABEL_MAP.get(
            result["label"],
            result["label"]
        ),
        "score": float(
            result["score"]
        ),
    }


# =====================================================
# OPTIONAL OPENAI / LOCAL LLM SUPPORT
# =====================================================

class LLMClient(Protocol):

    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class OpenAIChatCompletionsClient:

    api_key: str

    model: str = "gpt-4o-mini"

    timeout_seconds: float = 20.0

    def complete(self, prompt: str) -> str:

        url = (
            "https://api.openai.com/v1/chat/completions"
        )

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content":
                    "You are a precise JSON-only incident diagnosis engine."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization":
                f"Bearer {self.api_key}",

                "Content-Type":
                "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req,
            timeout=self.timeout_seconds
        ) as resp:

            body = resp.read().decode(
                "utf-8"
            )

        parsed = json.loads(body)

        return parsed["choices"][0]["message"]["content"]


def build_llm_from_env() -> Optional[LLMClient]:

    api_key = os.getenv(
        "OPENAI_API_KEY",
        ""
    ).strip()

    if not api_key:

        return None

    model = os.getenv(
        "AGENTCLOUD_LLM_MODEL",
        "gpt-4o-mini"
    ).strip() or "gpt-4o-mini"

    timeout = float(
        os.getenv(
            "AGENTCLOUD_LLM_TIMEOUT_SECONDS",
            "20"
        )
    )

    return OpenAIChatCompletionsClient(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout
    )


# =====================================================
# QWEN LOCAL REASONING
# =====================================================

_qwen_tokenizer = None

_qwen_model = None


def load_qwen():

    global _qwen_tokenizer
    global _qwen_model

    if _qwen_model is None:

        print(
            "[LLM] Loading Qwen..."
        )

        model_name = (
            "Qwen/Qwen2.5-0.5B-Instruct"
        )

        _qwen_tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        _qwen_model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        )

    return (
        _qwen_tokenizer,
        _qwen_model
    )


def generate_plan_with_qwen(
    incident: str,
    severity: str,
):

    tokenizer, model = load_qwen()

    prompt = f"""
You are an autonomous cloud incident response planner.

Incident:
{incident}

Severity:
{severity}

Return ONLY valid JSON.

Example:
{{
  "action": "restart_service",
  "target": "lb"
}}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.2
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(
        "[QWEN]",
        text
    )

    try:

        import re

        match = re.search(
            r'\{[\s\S]*?\}',
            text
        )

        if not match:
            return None

        json_text = match.group(0)

        json_text = text[
            start:end
        ]

        return json.loads(
            json_text
        )

    except Exception:

        return None