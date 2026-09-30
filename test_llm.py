from openai import OpenAI

from src.config import settings

client = OpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key.get_secret_value(),
)

resp = client.chat.completions.create(
    model=settings.llm_model,
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)