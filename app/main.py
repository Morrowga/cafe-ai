"""
app/main.py

CaféAI — FastAPI entry point.
Run with: uvicorn app.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import HealthResponse
from app.routers.recipe import router as recipe_router

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan: warm up HF model on startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("☕ CaféAI starting up...")
    logger.info("✅ Server is hot.")
    yield
    logger.info("CaféAI shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CaféAI API",
    description="Generate personalized coffee recipes from your mood using HuggingFace + LangChain + Claude",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev
        "http://localhost:5173",   # Vite dev
        "https://thihaeung.vercel.app",
        "https://thihaeung.com",
        "https://www.thihaeung.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(recipe_router)


# ── Root ───────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "hf_model": "facebook/bart-large-mnli (Inference API)",
        "llm_model": "gpt-4o-mini",
    }