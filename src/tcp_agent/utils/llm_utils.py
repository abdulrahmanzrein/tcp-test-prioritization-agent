
from __future__ import annotations

import os
import time
import logging
from typing import Any
from tcp_agent.utils.token_logger import token_logger

logger = logging.getLogger(__name__)

def resolve_provider(model_name: str) -> str | None:
    name = model_name.lower()
    if "gemini" in name:
        return "google_genai"
    if name.startswith("claude"):
        return "anthropic"
    if "mistral" in name:
        return "mistralai"
    # Alibaba Qwen and local OpenAI-compat tags (e.g. Ollama "qwen2.5:32b") use ChatOpenAI.
    if "qwen" in name:
        return "openai"
    if name.startswith("o1") or name.startswith("o3"):
        return "openai"
    return None


def openai_compat_base_from_env() -> str:
    """Base URL for OpenAI-compatible backends (Ollama, vLLM, etc.). Include /v1 suffix."""
    return (
        os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
    ).strip()


def uses_openai_sdk_stack(model_name: str) -> bool:
    """True if LangChain routes this model through ChatOpenAI (incl. Qwen via compat API)."""
    return resolve_provider(model_name) in (None, "openai")


def build_init_chat_model_kwargs(model_name: str, *, skip_temperature: bool) -> dict[str, Any]:
    """Extra kwargs for init_chat_model: temperature + optional OpenAI client base URL / key."""
    kw: dict[str, Any] = {}
    if not skip_temperature:
        kw["temperature"] = 0
    if not uses_openai_sdk_stack(model_name):
        return kw
    base = openai_compat_base_from_env()
    if base:
        kw["base_url"] = base.rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        kw["api_key"] = api_key
    elif base:
        # Ollama and many compat servers ignore the key; SDK still expects a non-empty string.
        kw["api_key"] = "ollama"
    return kw

def invoke_with_retry(model: Any, messages: list, model_name: str, max_retries: int = 6):
    """Common retry logic with token usage logging."""
    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)
            
            # Log usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                token_logger.log_request(
                    model_name=model_name,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    status="success"
                )
            
            return response
            
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()
            
            # Log failure
            token_logger.log_request(
                model_name=model_name,
                input_tokens=0,
                output_tokens=0,
                status=f"error: {name}"
            )
            
            if name == "LengthFinishReasonError" or "length limit" in msg:
                raise
            if "401" in str(e) or "403" in str(e) or "invalid_api_key" in msg:
                raise
            if attempt == max_retries - 1:
                raise
            
            # Rate limit backoff
            wait = 65 if "rate" in msg or "429" in str(e) else min(2 ** attempt, 30)
            print(f"  [{model_name}-RETRY] {attempt + 1}/{max_retries} {name}: {str(e)[:120]}", flush=True)
            time.sleep(wait)
