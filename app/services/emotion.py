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
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

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

    # Log raw response for debugging
    if res.status_code != 200:
        raise ValueError(f"HF API returned {res.status_code}: {res.text}")

    result = res.json()

    if isinstance(result, list):
        result = result[0]  # ← unwrap list

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