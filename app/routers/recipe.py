"""
routers/recipe.py

POST /api/generate-recipe

Pipeline:
  1. Input validation
     - OpenAI Moderation API  → block harmful content
     - GPT relevance check    → block non-coffee/mood input
  2. Emotion detection — HuggingFace Inference API
  3. Recipe generation — LangChain + GPT (retry up to 3x on bad JSON)
  4. Return structured recipe
"""

import os
import httpx
import logging
import re

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.models.schemas import RecipeRequest, RecipeResponse
from app.services.emotion import detect_emotions
from app.services.recipe import generate_recipe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recipe"])

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Layer 1: Quick gibberish check (free, instant) ─────────────────────────

def is_meaningful_text(text: str) -> bool:
    """
    Rejects obvious gibberish before wasting any API calls.
    """
    text = text.strip()

    if len(text) < 3:
        return False

    # Repeated character patterns like "hehehehe", "aaaaaaa"
    if re.match(r'^(.{1,3})\1{3,}$', text, re.IGNORECASE):
        return False

    # Only numbers or special characters
    if re.match(r'^[\d\W]+$', text):
        return False

    return True


# ── Layer 2: OpenAI Moderation API (harmful content) ──────────────────────

async def is_safe_content(text: str) -> bool:
    """
    Use OpenAI Moderation API to detect harmful content.
    Free to use — no extra cost on top of your OpenAI account.
    """
    try:
        response = await openai_client.moderations.create(input=text)
        flagged = response.results[0].flagged
        if flagged:
            logger.info(f"Moderation flagged: '{text[:40]}'")
        return not flagged
    except Exception as e:
        logger.warning(f"Moderation API error: {e} — allowing through")
        return True  # allow through if moderation API fails


# ── Layer 3: GPT relevance check (coffee/mood related?) ───────────────────

async def is_coffee_or_mood_related(text: str) -> bool:
    """
    Use GPT to determine if input is related to coffee or mood.
    GPT understands intent — not just keywords.

    Examples:
      "give me a knife"   → no  → blocked
      "I'm exhausted"     → yes → allowed
      "no sugar please"   → yes → allowed
      "give me a sword"   → no  → blocked
      "I need a weapon"   → no  → blocked
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Is this message related to any of the following?
- Coffee, drinks, or café orders (e.g. "make me a latte", "hot coffee please", "no sugar")
- Mood, feelings, or emotions (e.g. "I'm tired", "feeling stressed", "need energy")
- Physical states that relate to needing coffee (e.g. "I can't wake up", "long day")

Answer only 'yes' or 'no'. Nothing else.
Message: "{text}"
"""
            }],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        logger.info(f"Relevance check: '{text[:40]}' → {answer}")
        return answer == "yes"
    except Exception as e:
        logger.warning(f"Relevance check error: {e} — allowing through")
        return True


# ── Main endpoint ──────────────────────────────────────────────────────────

@router.post("/generate-recipe", response_model=RecipeResponse)
async def create_recipe(request: RecipeRequest):
    """
    Generate a personalized coffee recipe from mood input.
    """

    # ── Step 1a: Empty input check ─────────────────────────────────────────
    if not request.text.strip() and not request.tags:
        raise HTTPException(
            status_code=422,
            detail="Please tell me how you're feeling or select a mood tag. ☕"
        )

    if request.text.strip():

        # ── Step 1b: Gibberish check (instant, free) ───────────────────────
        if not is_meaningful_text(request.text):
            raise HTTPException(
                status_code=422,
                detail="I can only make coffee, that's all. ☕"
            )

        # ── Step 1c: Harmful content check (OpenAI Moderation) ────────────
        if not await is_safe_content(request.text):
            raise HTTPException(
                status_code=422,
                detail="I can only make coffee, that's all. ☕"
            )

        # ── Step 1d: Relevance check (GPT) ────────────────────────────────
        if not await is_coffee_or_mood_related(request.text):
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
            break
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