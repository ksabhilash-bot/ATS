"""
Tier 3 of the hybrid pipeline: LLM-based section classification.

Only called for blocks that survived both:
  1. Rule-based regex (utils/section_classifier.py)      -> UNKNOWN
  2. NLP/TF-IDF similarity (utils/nlp_section_classifier.py) -> not confident

This keeps LLM calls rare (and cheap) instead of classifying every
block with the LLM, which would be slow and unnecessary for resumes
with normal headings.

Uses the same Mistral model already wired into the project
(Services/candidateResumeLLMService.py) via langchain_mistralai.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from utils.section_classifier import ResumeSection

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

_llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-large-latest",
    temperature=0,
)


class _SectionLabel(BaseModel):
    section: str = Field(
        description=(
            "One of: contact, summary, skills, work_experience, education, "
            "projects, certifications, awards, languages, volunteer, unknown. "
            "Pick 'unknown' if the text genuinely doesn't fit any category."
        )
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


_SYSTEM_PROMPT = """You are a precise resume-section classifier for an ATS system.
You will be given a single block of text extracted from a resume (no heading,
or an unrecognized heading). Classify which resume section this content
belongs to. Choose exactly one of the allowed labels. Do not invent new labels."""

_VALID_VALUES = {s.value for s in ResumeSection}


async def classify_block_llm(text: str) -> tuple[ResumeSection, str]:
    """
    Ask the LLM to classify a single ambiguous text block.
    Returns (ResumeSection, reasoning). Falls back to UNKNOWN on any
    parsing issue rather than raising, since this is a best-effort tier.
    """
    if not text.strip():
        return ResumeSection.UNKNOWN, "empty block"

    structured_llm = _llm.with_structured_output(_SectionLabel)

    try:
        result: _SectionLabel = await structured_llm.ainvoke(
            f"{_SYSTEM_PROMPT}\n\nTEXT BLOCK:\n{text.strip()[:1500]}"
        )
    except Exception as error:  # noqa: BLE001 - best-effort tier, never crash pipeline
        return ResumeSection.UNKNOWN, f"llm_error: {error}"

    label = result.section.strip().lower()
    if label not in _VALID_VALUES:
        return ResumeSection.UNKNOWN, f"invalid_label_returned: {label}"

    return ResumeSection(label), result.reasoning