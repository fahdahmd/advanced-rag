import time
from functools import lru_cache

from openai import OpenAI, PermissionDeniedError, RateLimitError

from src.config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key.get_secret_value(),
        timeout=120,
    )



class FatalLLMError(RuntimeError):
    """An error where retrying, or continuing the run, is pointless."""
    
def chat(messages: list[dict], max_tokens: int = 2000, retries: int = 4) -> str:
    """One LLM call with retries. temperature=0 keeps eval runs repeatable."""
    last_error = "unknown"
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=settings.llm_model, messages=messages, temperature=0, max_tokens=max_tokens
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            last_error = "empty response"
        except PermissionDeniedError as e:
            raise FatalLLMError(f"Access denied: {e}") from e
        except RateLimitError as e:
            if "per-day" in str(e):
                raise FatalLLMError("Daily request limit reached. Add credits or wait for the reset.") from e
            last_error = f"RateLimitError: {e}"  # a short-term limit, so waiting helps
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(3 * 2**attempt)
    raise RuntimeError(f"LLM call failed after {retries} tries: {last_error}")