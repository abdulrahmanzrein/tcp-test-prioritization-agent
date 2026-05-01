
import time
import logging
from typing import Any
from langchain_core.messages import BaseMessage
from tcp_agent.utils.token_logger import token_logger

logger = logging.getLogger(__name__)

def resolve_provider(model_name: str) -> str | None:
    name = model_name.lower()
    if "gemini" in name:
        return "google_genai"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("o1") or name.startswith("o3"):
        return "openai"
    return None

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
