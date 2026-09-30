from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Secrets
    openrouter_api_key: SecretStr | None = None

    # SEC requires a real name + email in every request
    sec_user_agent: str

    # Paths
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    chroma_dir: Path = ROOT / "chroma_db"

    # LLM (via OpenRouter)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"  # verify the ID on openrouter.ai/models

    # Local models (unchanged)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"


settings = Settings()