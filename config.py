from pathlib import Path

from argue_eval.utils import (  # Internal evaluation framework
    DEFAULT_MODELS_BY_PROVIDER,
    ModelProvider,
)

DATA_DIR = Path("<PATH_TO_DOCUMENTS>")
REQUESTS_PATH = Path("<PATH_TO_TOPICS>")

# for step 1
QA_PATH = Path("grounded_questions.json")

# for step 2
PARAPHRASE_JUDGMENTS_NAME = "paraphrase_judgments.json"
MERGED_OUTPUT_DIR = Path("merged_questions")


STOP_SEQUENCES = ["<|eot_id|>"]

PROVIDER = ModelProvider.OPENAI  # Replace with appropriate provider
MODEL_NAME = DEFAULT_MODELS_BY_PROVIDER[PROVIDER]
