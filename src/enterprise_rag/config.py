"""
Configuration for Enterprise RAG system.

Uses pydantic-settings with nested configuration groups.
"""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file FIRST - sets env vars for external SDKs (OpenAI, etc.)
load_dotenv()


# =============================================================================
# Nested Configuration Groups
# =============================================================================


class LangSmithConfig(BaseSettings):
    """LangSmith observability settings."""

    model_config = SettingsConfigDict(env_prefix="LANGSMITH_", extra="ignore")

    TRACING: str = "false"
    ENDPOINT: str = "https://api.smith.langchain.com"
    API_KEY: str = ""
    PROJECT: str = ""


class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI settings."""

    model_config = SettingsConfigDict(env_prefix="AZURE_", extra="ignore")

    OPENAI_ENDPOINT: str = ""
    OPENAI_API_KEY: str = ""
    EMBEDDING_DEPLOYMENT_NAME: str = ""
    EMBEDDING_API_VERSION: str = "2024-02-01"
    OPENAI_API_VERSION: str = "2025-03-01-preview"


class PostgresConfig(BaseSettings):
    """PostgreSQL settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    DB_URI: str = ""
    DB_URI_POOLER: str = ""
    HOSTNAME: str = ""


class ElasticsearchConfig(BaseSettings):
    """Elasticsearch settings."""

    model_config = SettingsConfigDict(extra="ignore")

    ELASTICSEARCH_URL: str = Field(default="", alias="ELASTICSEARCH_URL")
    ELASTICSEARCH_API_KEY: str = Field(default="", alias="ELASTICSEARCH_API_KEY")

    # Vector indices
    WIKI_ES_VECTOR_INDEX: str = "elasticgpt-embeddings-wiki"
    SNOW_ES_VECTOR_INDEX: str = "test-bq-embeddings-openai"
    JIRA_ES_VECTOR_INDEX: str = "elasticgpt-jira-embeddings-dev"
    DOCS_ES_VECTOR_INDEX: str = "elasticgpt-embeddings-docs"

    # Retrieval parameters
    ES_K: int = 5
    ES_RANK_CONSTANT: int = 20


class LLMConfig(BaseSettings):
    """LLM model settings."""

    model_config = SettingsConfigDict(extra="ignore")

    CLASSIFIER_MODEL: str = "gpt-5-nano"
    CLASSIFIER_TEMPERATURE: float = 0.0
    CLASSIFIER_REASONING_EFFORT: str = "low"
    RESPONSE_MODEL: str = "gpt-5-nano"
    RESPONSE_TEMPERATURE: float = 0.3
    RESPONSE_REASONING_EFFORT: str = "low"
    TIMEOUT_SECONDS: int = 90


class EvalConfig(BaseSettings):
    """Evaluation settings."""

    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="ignore")

    AGENT_MODEL: str = "gpt-4.1-mini"
    JUDGE_MODEL: str = "gemini-3-flash-preview"


class JinaConfig(BaseSettings):
    """Jina AI settings for reranking."""

    model_config = SettingsConfigDict(env_prefix="JINA_", extra="ignore")

    API_KEY: str = ""
    RERANK_MODEL: str = "jina-reranker-v3"
    RERANK_TOP_N: int = 8  # Final output after reranking


class RetrievalConfig(BaseSettings):
    """Retrieval settings."""

    model_config = SettingsConfigDict(extra="ignore")

    # Candidate retrieval (before reranking)
    CANDIDATE_K: int = 15  # Retrieve per source, rerank to few

    # Legacy (kept for compatibility)
    MAX_CHUNKS_PER_AGENT: int = 5
    MIN_RELEVANCE_SCORE: float = 0.5


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """
    Application settings with nested configuration groups.

    Usage:
        from enterprise_rag.config import settings

        settings.azure.OPENAI_API_KEY     # Azure OpenAI
        settings.azure.OPENAI_ENDPOINT    # Azure endpoint
        settings.llm.CLASSIFIER_MODEL     # Model settings
        settings.elasticsearch.ELASTICSEARCH_URL
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # Top-level Settings
    # =========================================================================

    GOOGLE_API_KEY: str = ""
    DEBUG: bool = False

    # =========================================================================
    # Nested config groups
    # =========================================================================

    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    azure: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)


# Global settings instance
settings = Settings()
