from dotenv import load_dotenv

# Load .env into os.environ so libraries that read env vars directly
# (LangSmith, LangChain) can see our configuration. pydantic-settings below
# reads the same file into the typed Settings object for our own code.
load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, loaded from env vars / local .env."""

    # --- Database ---
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- OpenAI ---
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # must match the vector(N) column below
    reasoning_effort: str = "none"  # gpt-5.x: tools via chat.completions need 'none'

    # --- SEC EDGAR ---
    edgar_identity: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def database_url(self) -> str:
        # Cloud Run mounts Cloud SQL as a unix socket at /cloudsql/<conn-name>.
        # libpq reads the socket directory from the `host` query parameter.
        if self.postgres_host.startswith("/"):
            return (
                f"postgresql://{self.postgres_user}:{self.postgres_password}@/"
                f"{self.postgres_db}?host={self.postgres_host}"
            )
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    chat_model: str = "gpt-5.6-terra"  # swap in .env if this ID errors
    retrieval_top_k: int = 5

    # --- Web search ---
    tavily_api_key: str = ""

settings = Settings()