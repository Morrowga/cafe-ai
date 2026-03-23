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

from fastapi import APIRouter, HTTPException

from app.models.schemas import RecipeRequest, RecipeResponse
from app.services.emotion import detect_emotions, HF_API_URL
from app.services.recipe import generate_recipe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recipe"])


# ── Topic labels ───────────────────────────────────────────────────────────
TOPIC_LABELS = [
    "the person is describing their mood, feeling, or emotion and wants a coffee recommendation",
    "the person is making a specific coffee request or describing coffee preferences",
    "the person is asking for something completely unrelated to coffee, mood, or feelings",
]


async def is_coffee_or_mood_related(text: str) -> bool:
    """
    Use HuggingFace Inference API to check if input is coffee/mood related.
    Returns True if allowed, False if should be blocked.
    """
    async with httpx.AsyncClient() as client:
        res = await client.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
            json={
                "inputs": text,
                "parameters": {"candidate_labels": TOPIC_LABELS},
            },
            timeout=30.0,
        )
        result = res.json()

    if isinstance(result, dict) and result.get("error"):
        # If HF API has an error, allow through and let emotion detection handle it
        logger.warning(f"HF topic check error: {result['error']}")
        return True

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    logger.info(f"Topic check → label: '{top_label[:40]}' score: {top_score:.2f}")

    return top_label != TOPIC_LABELS[2] and top_score > 0.55


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