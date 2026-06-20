import os
import yaml
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _load_yaml(path: str) -> dict:
    with open(os.path.join(os.path.dirname(__file__), path)) as f:
        return yaml.safe_load(f)


AGENTS_CONFIG = _load_yaml("config/agents.yaml")
TASKS_CONFIG = _load_yaml("config/tasks.yaml")

MODEL_MAP = {
    "claim_extractor_model": os.getenv("MODEL_CLAIM_EXTRACTOR"),
    "visual_judge_model": os.getenv("MODEL_VISUAL_JUDGE"),
    "quality_judge_model": os.getenv("MODEL_QUALITY_JUDGE"),
    "authenticity_judge_model": os.getenv("MODEL_AUTHENTICITY_JUDGE"),
    "synthesizer_model": os.getenv("MODEL_SYNTHESIZER"),
}

MODEL_BASE_URL_MAP = {
    "gemma4": os.getenv("OLLAMA_MODEL_GEMMA4_URL", "http://localhost:11434"),
    "gemma3": os.getenv("OLLAMA_MODEL_GEMMA3_URL", "http://localhost:11434"),
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output.csv")

for agent_key, agent_cfg in AGENTS_CONFIG.items():
    llm_key = agent_cfg.get("llm", "")
    if llm_key in MODEL_MAP:
        agent_cfg["llm"] = MODEL_MAP[llm_key]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
