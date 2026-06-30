"""
Hybrid resume section classifier — orchestrates all three tiers.

Pipeline:
    1. Rule-based regex headings  (utils/section_classifier.py)
    2. NLP / TF-IDF similarity    (utils/nlp_section_classifier.py)
    3. LLM classification         (utils/llm_section_fallback.py)

Only blocks that fail a tier escalate to the next one, so most resumes
(clean headings) never touch tiers 2/3 and stay fast + free.

This is the function the rest of the app (candidateResumeLLMService.py
etc.) should call going forward instead of calling classify_sections()
directly, if you want the upgraded hybrid behavior. The original
classify_sections() is untouched/still usable on its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from utils.llm_section_fallback import classify_block_llm
from utils.nlp_section_classifier import classify_block_nlp, is_confident
from utils.section_classifier import (
    ResumeSection,
    SectionBlock,
    classify_sections,
)


async def classify_sections_hybrid(
    clean_text: str,
    use_llm_fallback: bool = True,
) -> list[SectionBlock]:
    """
    Full hybrid classification pipeline.

    Args:
        clean_text: cleaned resume text (output of utils.text_cleaner).
        use_llm_fallback: set False to stop at tier 2 (NLP) and skip
            LLM API calls entirely - useful for tests/cost control.

    Returns:
        List of SectionBlock with confidence in {"rule", "nlp", "llm", "inferred"}.
    """
    blocks = classify_sections(clean_text)
    resolved: list[SectionBlock] = []

    for block in blocks:
        # Only a genuine regex heading match should skip tiers 2/3.
        # NOTE: classify_sections() has a post-process heuristic that
        # relabels the first UNKNOWN block as CONTACT (confidence=
        # "inferred") when no headings were found at all anywhere in the
        # resume. That's a positional guess, not a real classification -
        # if we gated on `block.section != UNKNOWN` instead of on
        # confidence, a resume with zero detected headings would have its
        # *entire* text dumped into one "contact" block and never reach
        # the NLP/LLM tiers at all. Gating on confidence=="rule" avoids
        # that trap.
        if block.confidence == "rule":
            resolved.append(block)
            continue

        # Tier 2: NLP similarity
        nlp_pred = classify_block_nlp(block.content)

        if is_confident(nlp_pred):
            resolved.append(replace(
                block,
                section=nlp_pred.section,
                confidence="nlp",
            ))
            continue

        # Tier 3: LLM (only reached for genuinely ambiguous blocks)
        if use_llm_fallback:
            llm_section, _reasoning = await classify_block_llm(block.content)
            resolved.append(replace(
                block,
                section=llm_section,
                confidence="llm" if llm_section != ResumeSection.UNKNOWN else "inferred",
            ))
        else:
            # Keep the NLP best-guess even if low-confidence, rather than
            # leaving it fully UNKNOWN, but flag it clearly.
            resolved.append(replace(
                block,
                section=nlp_pred.section if nlp_pred.score > 0 else ResumeSection.UNKNOWN,
                confidence="nlp_low_confidence",
            ))

    return resolved


def classify_sections_hybrid_sync(
    clean_text: str,
    use_llm_fallback: bool = True,
) -> list[SectionBlock]:
    """Sync convenience wrapper for non-async callers."""
    return asyncio.run(classify_sections_hybrid(clean_text, use_llm_fallback))