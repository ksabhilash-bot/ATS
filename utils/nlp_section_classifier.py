"""
NLP-based fallback section classifier.

Tier 2 of the hybrid pipeline: only runs on blocks the rule-based
regex classifier (utils/section_classifier.py) could NOT label.

Approach: TF-IDF + cosine similarity against a small canonical corpus
of example content for each ResumeSection. No training data required,
no heavy model download, works offline, costs nothing per call.

Why TF-IDF over spaCy/embeddings here:
- Resume section *content* (not headings) is highly keyword-dense
  ("Python, FastAPI, Docker" / "Bachelor of Technology, CGPA 8.2")
  so sparse keyword vectors separate sections well without needing
  semantic embeddings.
- No extra heavyweight dependency (spaCy model ~50MB+) for a problem
  that doesn't need deep semantics.
- Fast enough to run per-block with no batching/GPU concerns.

Why this isn't the *first* tier: it's slower and fuzzier than regex
heading matches, and less precise than the rule pass on clean resumes.
It exists purely to catch what regex misses (no heading line, heading
inside a table cell, non-standard phrasing like "My Skill Set").
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.section_classifier import ResumeSection

# ── Canonical reference corpus ────────────────────────────────────────────
# Each entry is representative CONTENT (not headings) for that section.
# Used to build per-section TF-IDF centroids. Extend these as you label
# more real resumes and notice misclassifications.

_REFERENCE_CORPUS: dict[ResumeSection, str] = {
    ResumeSection.CONTACT: """
        email phone number address linkedin github portfolio website
        city state country mobile contact number
    """,
    ResumeSection.SUMMARY: """
        results driven motivated passionate experienced professional
        seeking opportunity to leverage skills years of experience
        proven track record dedicated detail oriented team player
    """,
    ResumeSection.SKILLS: """
        python java javascript typescript react node fastapi django
        sql postgresql mongodb docker kubernetes aws azure git
        machine learning data analysis communication leadership
        problem solving frameworks libraries tools technologies proficient
    """,
    ResumeSection.WORK_EXPERIENCE: """
        company position role responsible for managed led developed
        implemented designed built collaborated with team increased
        reduced improved delivered software engineer developer
        intern from to present january february march duties included
    """,
    ResumeSection.EDUCATION: """
        bachelor master degree university college school cgpa gpa
        percentage graduated major minor coursework thesis class of
        computer science engineering b.tech m.tech mca bca diploma
    """,
    ResumeSection.PROJECTS: """
        project built developed created application system using
        technologies github repository demo live link features
        implemented architecture deployed open source contribution
    """,
    ResumeSection.CERTIFICATIONS: """
        certified certificate completed course issued by valid
        credential id certification authority training program
        udemy coursera aws certified google certified microsoft
    """,
    ResumeSection.AWARDS: """
        award winner first place recognized for outstanding achievement
        honor scholarship competition hackathon rank top performer
    """,
    ResumeSection.LANGUAGES: """
        english hindi spanish french fluent native proficient
        intermediate beginner spoken written language proficiency
    """,
    ResumeSection.VOLUNTEER: """
        volunteer ngo community service organized event charity
        non profit social work outreach program coordinator
    """,
}

_SECTIONS = list(_REFERENCE_CORPUS.keys())

# Fit once at import time on the reference corpus
_VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_REFERENCE_MATRIX = _VECTORIZER.fit_transform(
    [_REFERENCE_CORPUS[s] for s in _SECTIONS]
)


@dataclass
class NLPPrediction:
    section: ResumeSection
    score: float          # cosine similarity, 0..1
    runner_up: ResumeSection
    runner_up_score: float


# Below this similarity, NLP tier is not confident -> escalate to LLM tier
NLP_CONFIDENCE_THRESHOLD = 0.12

# If the gap between best and second-best is too small, the block is
# genuinely ambiguous (could be Projects vs Work Experience etc.) -> escalate
NLP_MIN_MARGIN = 0.03


def classify_block_nlp(text: str) -> NLPPrediction:
    """
    Classify a single unlabeled text block against the reference corpus.
    Returns the best + runner-up section with similarity scores so the
    caller can decide whether to trust this or escalate further.
    """
    if not text.strip():
        return NLPPrediction(ResumeSection.UNKNOWN, 0.0, ResumeSection.UNKNOWN, 0.0)

    vec = _VECTORIZER.transform([text])
    sims = cosine_similarity(vec, _REFERENCE_MATRIX)[0]

    ranked = sorted(zip(_SECTIONS, sims), key=lambda x: x[1], reverse=True)
    best_section, best_score = ranked[0]
    runner_section, runner_score = ranked[1]

    return NLPPrediction(
        section=best_section,
        score=float(best_score),
        runner_up=runner_section,
        runner_up_score=float(runner_score),
    )


def is_confident(pred: NLPPrediction) -> bool:
    """Decide whether the NLP-tier prediction is trustworthy enough to use
    without escalating to the (expensive) LLM tier."""
    return (
        pred.score >= NLP_CONFIDENCE_THRESHOLD
        and (pred.score - pred.runner_up_score) >= NLP_MIN_MARGIN
    )