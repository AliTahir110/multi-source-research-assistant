import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

def get_env(name: str) -> str:
    """
    Fetch an environment variable or fail fast with a clear error.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

# Required secrets
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
TAVILY_API_KEY = get_env("TAVILY_API_KEY")

# Optional config (safe default)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
