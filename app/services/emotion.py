"""
services/emotion.py

Hugging Face Inference API — zero-shot classification.

Changed from local model to HF Inference API:
  - No torch/transformers needed on EC2
  - HuggingFace runs the model on their GPU servers
  - Your EC2 just makes a simple HTTP call
  - Response in 1-3 seconds instead of 30-90 seconds
"""

import os
import httpx

# ── HuggingFace Inference API ──────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

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


async def detect_emotions(text: str, top_n: int = 3) -> dict:
    """
    Run zero-shot classification via HuggingFace Inference API.

    Args:
        text  : User's free-form mood description
        top_n : How many top emotions to return (default 3)

    Returns:
        dict of { emotion: confidence_score }
        e.g. { "tired": 0.91, "stressed": 0.78, "cozy": 0.42 }
    """
    async with httpx.AsyncClient() as client:
        res = await client.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
            json={
                "inputs": text,
                "parameters": {"candidate_labels": EMOTION_LABELS},
            },
            timeout=30.0,
        )
        result = res.json()

    # Handle model loading state — HF sometimes returns 503 while warming up
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(f"HuggingFace API error: {result['error']}")

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