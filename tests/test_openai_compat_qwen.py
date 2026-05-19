"""OpenAI-compat (Ollama/vLLM) + Qwen routing via llm_utils."""
from __future__ import annotations

import pytest

from tcp_agent.utils.llm_utils import (
    build_init_chat_model_kwargs,
    resolve_provider,
    uses_openai_sdk_stack,
)


def test_qwen_maps_to_openai_provider():
    assert resolve_provider("qwen2.5:32b") == "openai"
    assert uses_openai_sdk_stack("qwen2.5:32b") is True


def test_gemini_not_openai_sdk_stack():
    assert uses_openai_sdk_stack("gemini-2.0-flash") is False


@pytest.mark.parametrize("skip_temperature", [True, False])
def test_build_kwargs_injects_placeholder_key_when_only_base_url(
    monkeypatch: pytest.MonkeyPatch, skip_temperature: bool
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1/")
    kw = build_init_chat_model_kwargs("qwen2.5:32b", skip_temperature=skip_temperature)
    assert kw["base_url"] == "http://127.0.0.1:11434/v1"
    assert kw["api_key"] == "ollama"
    if skip_temperature:
        assert "temperature" not in kw
    else:
        assert kw["temperature"] == 0


def test_build_kwargs_gemini_ignores_openai_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    kw = build_init_chat_model_kwargs("gemini-2.0-flash", skip_temperature=False)
    assert "base_url" not in kw
    assert "api_key" not in kw
