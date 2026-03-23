"""
routers/recipe.py

POST /api/generate-recipe

Pipeline:
  1. Input validation  — reject nonsense / off-topic text (HF Inference API)
  2. Emotion detection — HuggingFace Inference API
  3. Recipe generation — LangChain + GPT (retry up to 3x on bad JSON)
  4. Return structured recipe
"""

import os
import httpx
import logging
import re

from fastapi import APIRouter, HTTPException

from app.models.schemas import RecipeRequest, RecipeResponse
from app.services.emotion import detect_emotions, HF_API_URL
from app.services.recipe import generate_recipe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recipe"])


# ── Topic labels ───────────────────────────────────────────────────────────
TOPIC_LABELS = [
    "the person is clearly describing a human emotion, feeling, or mental state such as tired, stressed, happy, or anxious",
    "the person is clearly requesting a specific type of coffee drink with specific preferences like strength, temperature, or ingredients",
    "the person is typing nonsense, random words, or something completely unrelated to coffee or human emotions",
]

def is_meaningful_text(text: str) -> bool:
    """
    Quick pre-check before calling HuggingFace.
    Rejects obvious gibberish, repeated characters, too short, etc.
    """
    text = text.strip()

    # Too short
    if len(text) < 3:
        return False

    # Only repeated characters like "hehehehe", "aaaaaaa", "lololol"
    if re.match(r'^(.{1,3})\1{3,}$', text, re.IGNORECASE):
        return False

    # Only numbers or special characters
    if re.match(r'^[\d\W]+$', text):
        return False

    # Less than 2 unique words
    words = text.lower().split()
    if len(set(words)) < 2 and len(words) < 3:
        return False

    return True

async def is_coffee_or_mood_related(text: str) -> bool:
    for attempt in range(2):  # retry once on timeout
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    HF_API_URL,
                    headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
                    json={
                        "inputs": text,
                        "parameters": {"candidate_labels": TOPIC_LABELS},
                    },
                    timeout=60.0,
                )
            break
        except httpx.ReadTimeout:
            if attempt == 1:
                logger.warning("HF topic check timed out — allowing through")
                return True  # allow through if HF keeps timing out
            continue

    if res.status_code != 200:
        logger.warning(f"HF topic check failed {res.status_code}: {res.text}")
        return True

    result = res.json()

    if isinstance(result, dict) and result.get("error"):
        logger.warning(f"HF topic check error: {result['error']}")
        return True

    sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
    top_label = sorted_result[0]["label"]
    top_score = sorted_result[0]["score"]

    logger.info(f"Topic check → label: '{top_label[:40]}' score: {top_score:.2f}")
    return top_label != TOPIC_LABELS[2] and top_score > 0.60


@router.post("/generate-recipe", response_model=RecipeResponse)
async def create_recipe(request: RecipeRequest):
    """
    Generate a personalized coffee recipe from mood input.
    """

    # ── Step 1: Validate input ─────────────────────────────────────────────
    if not request.text.strip() and not request.tags:
        raise HTTPException(
            status_code=422,
            detail="Please tell me how you're feeling or select a mood tag. ☕"
        )

    # Pre-check for gibberish — before wasting an HF API call
    if request.text.strip() and not is_meaningful_text(request.text):
        raise HTTPException(
            status_code=422,
            detail="I can only make coffee, that's all. ☕"
        )

    # HuggingFace topic check
    if request.text.strip() and not await is_coffee_or_mood_related(request.text):
        raise HTTPException(
            status_code=422,
            detail="I can only make coffee, that's all. ☕"
        )


    logger.info(f"Recipe request: text='{request.text[:50]}' tags={request.tags}")

    # ── Step 2: HuggingFace emotion detection ──────────────────────────────
    try:
        detected_emotions = await detect_emotions(request.text)
        logger.info(f"Detected emotions: {detected_emotions}")
    except Exception as e:
        logger.error(f"HuggingFace error: {e}")
        raise HTTPException(status_code=500, detail="Emotion detection failed")

    # ── Step 3: LangChain + GPT recipe generation (retry up to 3x) ────────
    last_error = None
    recipe = None

    for attempt in range(3):
        try:
            recipe = await generate_recipe(
                user_text=request.text,
                detected_emotions=detected_emotions,
                mood_tags=request.tags,
            )
            break  # success — exit retry loop
        except ValueError as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1}/3 bad JSON: {e}")
            continue
        except Exception as e:
            logger.error(f"LLM hard error: {e}")
            raise HTTPException(status_code=500, detail="Recipe generation failed")

    if recipe is None:
        logger.error(f"All 3 attempts failed: {last_error}")
        raise HTTPException(
            status_code=500,
            detail="Recipe generation failed after 3 attempts. Please try again."
        )

    # ── Step 4: Attach emotion data and return ─────────────────────────────
    recipe["detected_emotions"] = detected_emotions
    return recipe