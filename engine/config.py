import os

from dotenv import load_dotenv

# ============ Configuration ============
load_dotenv()

class config:
    LLM_API_URL = os.getenv("LLM_API_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")

    PENDING_DIR = os.getenv("PENDING_DIR")
    PROCESSED_DIR = os.getenv("PROCESSED_DIR")
    RESULTS_DIR = os.getenv("RESULTS_DIR")
    LOG_DIR = os.getenv("LOG_DIR")

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_CHARACTERS = int(os.getenv("MAX_CHARACTERS", "5000"))
    MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))

    CLASSIFICATIONS_FILE = os.getenv("CLASSIFICATIONS_FILE")
    PROMPT_FILE = os.getenv("PROMPT_FILE")
