import time

from openai import OpenAI, PermissionDeniedError

from src.config import settings

CANDIDATES = [
    "thinkingmachines/inkling:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "qwen/qwen3.8-27b:free",
]

client = OpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key.get_secret_value(),
)

CONTEXT = (
    "Acme Corp 2023 10-K excerpt: Total revenue was $4.2 billion, "
    "up 12% from $3.75 billion in 2022. Net income was $610 million."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_filings",
        "description": "Search SEC filings for a query",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


def call(model, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(model=model, max_tokens=800, **kwargs)
            if resp.choices:
                return resp
            print(f"  empty response from {model}: {getattr(resp, 'error', None)}")
        except PermissionDeniedError as e:
            print(f"  403 for {model}: {e.message}")
            return None  # retrying a permission error never helps
        except Exception as e:
            print(f"  retry {attempt + 1} ({type(e).__name__}: {e})")
        time.sleep(5 * (attempt + 1))
    return None


def test_tool_call(model):
    resp = call(
        model,
        messages=[{"role": "user", "content": "What was Acme's revenue in 2023? Use the tool."}],
        tools=TOOLS,
    )
    return bool(resp and resp.choices[0].message.tool_calls)


def ask(model, question):
    resp = call(
        model,
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from the context. If the answer is not in the context, reply exactly: NOT_FOUND",
            },
            {"role": "user", "content": f"Context:\n{CONTEXT}\n\nQuestion: {question}"},
        ],
    )
    if not resp:
        return ""
    return resp.choices[0].message.content or ""


print(f"{'model':45} {'tools':6} {'grounded':9} {'not_found':10} {'secs':5}")
for model in CANDIDATES:
    start = time.time()
    tool_ok = test_tool_call(model)
    grounded = "4.2" in ask(model, "What was Acme's revenue in 2023?")
    refuses = "NOT_FOUND" in ask(model, "How many employees does Acme have?")
    secs = round(time.time() - start)
    print(f"{model:45} {tool_ok!s:6} {grounded!s:9} {refuses!s:10} {secs:<5}")