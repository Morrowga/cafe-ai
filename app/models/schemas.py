from pydantic import BaseModel, Field
from typing import List, Optional


class RecipeRequest(BaseModel):
    """
    Request body sent from the frontend.
    - text: free-form mood description (e.g. "I'm exhausted after a long day")
    - tags: mood chips selected by the user (e.g. ["tired", "need energy"])
    """
    text: str = Field(..., min_length=3, max_length=500, description="User's mood description")
    tags: Optional[List[str]] = Field(default=[], description="Selected mood tags")


class FlavorProfile(BaseModel):
    bitter: int = Field(..., ge=0, le=100)
    sweet:  int = Field(..., ge=0, le=100)
    acidic: int = Field(..., ge=0, le=100)
    creamy: int = Field(..., ge=0, le=100)
    bold:   int = Field(..., ge=0, le=100)


class RecipeResponse(BaseModel):
    """
    Full coffee recipe returned to the frontend.
    """
    recipe_name:       str
    tagline:           str
    flavor_story:      str
    beans:             str
    brew_method:       str
    ratio:             str
    temperature:       str
    steps:             List[str]
    flavor_profile:    FlavorProfile
    mood_match:        str

    # Hugging Face output — shown on frontend for portfolio transparency
    detected_emotions: dict


class HealthResponse(BaseModel):
    status: str
    hf_model: str
    llm_model: str