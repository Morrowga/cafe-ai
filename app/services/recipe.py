"""
services/recipe.py

LangChain orchestration layer.

Role in the pipeline:
  1. Receives structured emotion data from Hugging Face
  2. Extracts specific coffee preferences from user text (no sugar, no milk, etc.)
  3. Builds an enriched, context-aware prompt using PromptTemplate
  4. Sends to GPT via LCEL chain
  5. Parses and validates the structured JSON response
"""

import json
import os
import re
from functools import lru_cache

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.services.emotion import format_emotion_summary


# ── LLM Setup ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.9,
        max_tokens=1024,
    )


# ── Coffee preference extraction ───────────────────────────────────────────
#
# Pulls explicit preferences out of the user's text before passing to LLM.
# This ensures the LLM gets a clear, structured instruction to follow —
# not just buried in a freeform sentence it might ignore.
#
# Examples:
#   "give me no sugar one"       → "no sugar"
#   "I want something no milk"   → "no milk"
#   "strong and no ice please"   → "strong, no ice"
#   "I'm tired"                  → "none specified"
#
PREFERENCE_PATTERNS = [
    r"no\s+\w+",          # "no sugar", "no milk", "no ice"
    r"without\s+\w+",     # "without sugar"
    r"extra\s+\w+",       # "extra shot", "extra hot"
    r"less\s+\w+",        # "less bitter"
    r"more\s+\w+",        # "more creamy"
    r"strong(?:er)?",     # "strong", "stronger"
    r"weak(?:er)?",       # "weak", "weaker"
    r"hot(?:ter)?",       # "hot", "hotter"
    r"iced?",             # "ice", "iced"
    r"black",             # "black coffee"
    r"sweet(?:er)?",      # "sweet", "sweeter"
    r"decaf",             # "decaf"
    r"vegan",             # "vegan"
]

def extract_coffee_preferences(text: str) -> str:
    """
    Extract explicit coffee preferences from user text.
    Returns a comma-separated string of preferences, or 'none specified'.
    """
    if not text.strip():
        return "none specified"

    found = []
    text_lower = text.lower()
    for pattern in PREFERENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        found.extend(matches)

    if not found:
        return "none specified"

    # deduplicate while preserving order
    seen = set()
    unique = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return ", ".join(unique)


# ── Prompt Template ────────────────────────────────────────────────────────
#
# Variables injected at runtime:
#   {user_text}           - what the user typed
#   {emotion_summary}     - HuggingFace output, e.g. "tired (91%), stressed (78%)"
#   {mood_tags}           - chips user selected
#   {coffee_preferences}  - extracted preferences e.g. "no sugar, extra shot"
#
RECIPE_PROMPT = PromptTemplate(
    input_variables=["user_text", "emotion_summary", "mood_tags", "coffee_preferences"],
    template="""
You are CaféAI — a world-class barista, coffee poet, and mood reader.
You craft coffee recipes that feel deeply personal, not generic.

A customer walks in. Here is everything you know about them:

WHAT THEY SAID:
"{user_text}"

EMOTIONAL STATE (detected by AI analysis):
{emotion_summary}

MOOD TAGS THEY SELECTED:
{mood_tags}

SPECIFIC COFFEE PREFERENCES (follow these strictly, they are non-negotiable):
{coffee_preferences}

Based on ALL of this context, create a unique coffee recipe for them.
The recipe MUST respect the specific coffee preferences above.
Examples: if "no sugar" → never add sugar in the steps.
If "no milk" → black coffee only, no dairy or alternatives.
If "extra shot" → use double espresso.

Respond ONLY with a valid JSON object. No markdown. No explanation. No extra text.
Use this exact structure:

{{
  "recipe_name": "A poetic, creative name (not generic like 'Morning Espresso')",
  "tagline": "One evocative sentence that captures the feeling",
  "flavor_story": "2-3 sentences — a lifestyle story connecting their mood to this coffee. Make it atmospheric.",
  "beans": "Bean origin, variety, and roast level (e.g. Ethiopian Yirgacheffe, light roast)",
  "brew_method": "e.g. Pour Over, Moka Pot, Cold Brew, AeroPress, French Press",
  "ratio": "e.g. 1:15 coffee to water (20g coffee : 300ml water)",
  "temperature": "e.g. 93°C / 199°F",
  "steps": [
    "Step 1 — specific and instructional",
    "Step 2",
    "Step 3",
    "Step 4"
  ],
  "flavor_profile": {{
    "bitter": 0-100,
    "sweet": 0-100,
    "acidic": 0-100,
    "creamy": 0-100,
    "bold": 0-100
  }},
  "mood_match": "comma-separated list of the top detected moods"
}}
""",
)


# ── Chain ──────────────────────────────────────────────────────────────────

def get_recipe_chain():
    return RECIPE_PROMPT | get_llm() | StrOutputParser()


# ── Main function ──────────────────────────────────────────────────────────

async def generate_recipe(
    user_text: str,
    detected_emotions: dict,
    mood_tags: list[str],
) -> dict:
    """
    Full recipe generation pipeline:
      1. Extract coffee preferences from user text
      2. Format HF emotion data
      3. Run LangChain LCEL chain → GPT
      4. Parse and return JSON recipe
    """
    # Extract preferences BEFORE passing to LLM
    coffee_preferences = extract_coffee_preferences(user_text)

    emotion_summary = format_emotion_summary(detected_emotions)
    tags_str = ", ".join(mood_tags) if mood_tags else "none selected"

    chain = get_recipe_chain()
    raw_output = await chain.ainvoke({
        "user_text":          user_text,
        "emotion_summary":    emotion_summary,
        "mood_tags":          tags_str,
        "coffee_preferences": coffee_preferences,
    })

    clean = raw_output.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        recipe = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_output}")

    if not isinstance(recipe, dict):
        raise ValueError(f"LLM returned unexpected type: {type(recipe)}")

    return recipe