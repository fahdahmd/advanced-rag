import time
from functools import lru_cache

from openai import OpenAI, PermissionDeniedError

from src.config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key.get_secret_value(),
        timeout=120,
    )


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
        except PermissionDeniedError:
            raise  # retrying a 403 never helps
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(3 * 2**attempt)  # wait 3s, 6s, 12s... for overloaded free models
    raise RuntimeError(f"LLM call failed after {retries} tries: {last_error}")