"""
services/emotion.py

Hugging Face zero-shot classification.

Model : facebook/bart-large-mnli
Why   : No fine-tuning needed, runs on CPU, completely free.
        It classifies ANY text against ANY labels we define.

How it works:
  - We define our own emotion labels (tired, energetic, stressed, etc.)
  - The model reads the user's text and scores it against each label
  - Returns top N emotions with confidence percentages
  - These feed into LangChain as structured context for the recipe prompt
"""

from transformers import pipeline
from functools import lru_cache


# ── Emotion labels CaféAI understands ─────────────────────────────────────
EMOTION_LABELS = [
    "tired",
    "energetic",
    "stressed",
    "relaxed",
    "happy",
    "sad",
    "romantic",
    "focused",
    "adventurous",
    "cozy",
    "anxious",
    "motivated",
]


@lru_cache(maxsize=1)
def get_classifier():
    """
    Load the HuggingFace pipeline once and cache it.
    lru_cache ensures the model is loaded only on first call — not every request.
    device=-1 means CPU. Change to device=0 if you have a GPU.
    """
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )


def detect_emotions(text: str, top_n: int = 3) -> dict:
    """
    Run zero-shot classification on user input text.

    Args:
        text  : User's free-form mood description
        top_n : How many top emotions to return (default 3)

    Returns:
        dict of { emotion: confidence_score }
        e.g. { "tired": 0.91, "stressed": 0.78, "cozy": 0.42 }

    Example:
        detect_emotions("I'm exhausted after back-to-back meetings")
        → { "tired": 0.94, "stressed": 0.81, "anxious": 0.45 }
    """
    classifier = get_classifier()
    result = classifier(text, candidate_labels=EMOTION_LABELS)

    # result structure from HuggingFace:
    # { "sequence": "...", "labels": [...], "scores": [...] }
    top_emotions = dict(
        zip(result["labels"][:top_n], result["scores"][:top_n])
    )
    return top_emotions


def format_emotion_summary(emotions: dict) -> str:
    """
    Convert emotion dict to a readable string for the LangChain prompt.

    e.g. { "tired": 0.91, "stressed": 0.78 }
      → "tired (91%), stressed (78%)"
    """
    return ", ".join(
        f"{emotion} ({round(score * 100)}%)"
        for emotion, score in emotions.items()
    )