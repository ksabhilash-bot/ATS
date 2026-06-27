import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ResumeSection(str, Enum):
    CONTACT         = "contact"
    SUMMARY         = "summary"
    SKILLS          = "skills"
    WORK_EXPERIENCE = "work_experience"
    EDUCATION       = "education"
    PROJECTS        = "projects"
    CERTIFICATIONS  = "certifications"
    AWARDS          = "awards"
    LANGUAGES       = "languages"
    VOLUNTEER       = "volunteer"
    UNKNOWN         = "unknown"


@dataclass
class SectionBlock:
    section: ResumeSection
    heading: Optional[str]
    content: str
    confidence: str  # "rule" | "inferred"


# ── Heading keyword rules ─────────────────────────────────────────────────────

HEADING_RULES: list[tuple[ResumeSection, list[str]]] = [
    (ResumeSection.CONTACT, [
        r"contact(\s+information|\s+details?)?",
        r"personal\s+(info|details?|information)",
    ]),
    (ResumeSection.SUMMARY, [
        r"(professional\s+)?(summary|profile|objective)",
        r"about\s+(me|myself)",
        r"career\s+objective",
        r"executive\s+summary",
    ]),
    (ResumeSection.SKILLS, [
        r"(technical\s+|core\s+|key\s+)?skills?",
        r"technologies(\s+&\s+tools?)?",
        r"tools?(\s+&\s+technologies?)?",
        r"core\s+competenc(y|ies)",
        r"expertise",
        r"proficiencies",
        r"tech\s+stack",
    ]),
    (ResumeSection.WORK_EXPERIENCE, [
        r"(work\s+|professional\s+|relevant\s+)?experience",
        r"employment(\s+history)?",
        r"work\s+history",
        r"career\s+history",
        r"internship(s)?",
        r"positions?\s+held",
    ]),
    (ResumeSection.EDUCATION, [
        r"education(al)?(\s+background|\s+qualifications?)?",
        r"academic(s|\s+background|\s+qualifications?)?",
        r"degrees?",
        r"qualifications?",
        r"schooling",
    ]),
    (ResumeSection.PROJECTS, [
        r"(personal\s+|academic\s+|key\s+|notable\s+)?projects?",
        r"portfolio",
        r"notable\s+work",
        r"side\s+projects?",
    ]),
    (ResumeSection.CERTIFICATIONS, [
        r"certifications?",
        r"certificates?",
        r"licenses?(\s+(&|and)\s+certifications?)?",
        r"credentials?",
        r"accreditations?",
        r"courses?(\s+&\s+certifications?)?",
    ]),
    (ResumeSection.AWARDS, [
        r"awards?(\s+(&|and)\s+honors?)?",
        r"honors?(\s+(&|and)\s+awards?)?",
        r"achievements?",
        r"recognition",
        r"accomplishments?",
    ]),
    (ResumeSection.LANGUAGES, [
        r"languages?(\s+known|\s+spoken|\s+proficiency)?",
        r"linguistic\s+skills?",
    ]),
    (ResumeSection.VOLUNTEER, [
        r"volunteer(ing|\s+work|\s+experience)?",
        r"community\s+(service|involvement|work)",
        r"social\s+work",
        r"extra.?curricular(s|\s+activities?)?",
    ]),
]

# Pre-compile once at module load
_COMPILED: list[tuple[ResumeSection, list[re.Pattern]]] = [
    (sec, [re.compile(rf"^\s*{pat}\s*[:\-]?\s*$", re.IGNORECASE) for pat in pats])
    for sec, pats in HEADING_RULES
]

# Lines that look like a section heading: short, title/upper case, no sentence punctuation
_IS_HEADING = re.compile(
    r"^[A-Z][A-Za-z &/\(\)\-]{1,50}$"   # Title Case
    r"|^[A-Z &/\(\)\-]{2,50}$"            # ALL CAPS
)


def _match_heading(line: str) -> Optional[ResumeSection]:
    stripped = line.strip().rstrip(":").strip()
    for section, patterns in _COMPILED:
        for pat in patterns:
            if pat.match(stripped):
                return section
    return None


def classify_sections(clean_text: str) -> list[SectionBlock]:
    """
    Split cleaned resume text into labeled section blocks.

    Strategy:
    1. Walk lines top-to-bottom.
    2. If a short line matches a known heading keyword → start new section.
    3. If a short ALL CAPS / Title Case line doesn't match → UNKNOWN section.
    4. First UNKNOWN block at the top is re-labeled CONTACT (it's always contact info).
    5. Consecutive UNKNOWN blocks are merged.

    Returns:
        List of SectionBlock ordered as they appear in the resume.
    """
    lines = clean_text.splitlines()
    blocks: list[SectionBlock] = []

    current_section = ResumeSection.UNKNOWN
    current_heading: Optional[str] = None
    current_lines: list[str] = []
    current_start = 0

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append(SectionBlock(
                section    = current_section,
                heading    = current_heading,
                content    = content,
                confidence = "rule" if current_section != ResumeSection.UNKNOWN else "inferred",
            ))

    for line in lines:
        stripped = line.strip()

        if not stripped:
            current_lines.append("")
            continue

        # Only consider short lines (< 60 chars) as potential headings
        if len(stripped) < 60 and _IS_HEADING.match(stripped):
            detected = _match_heading(stripped)
            if detected is not None:
                flush()
                current_lines = []
                current_section = detected
                current_heading = stripped
                continue

        current_lines.append(line)

    flush()

    return _post_process(blocks)


def _post_process(blocks: list[SectionBlock]) -> list[SectionBlock]:
    """
    1. Merge consecutive UNKNOWN blocks.
    2. Tag first block as CONTACT if it's UNKNOWN (top of resume = contact info).
    """
    if not blocks:
        return blocks

    merged: list[SectionBlock] = []
    buf: Optional[SectionBlock] = None

    for block in blocks:
        if buf is None:
            buf = block
        elif buf.section == ResumeSection.UNKNOWN and block.section == ResumeSection.UNKNOWN:
            buf = SectionBlock(
                section    = ResumeSection.UNKNOWN,
                heading    = buf.heading,
                content    = buf.content + "\n" + block.content,
                confidence = "inferred",
            )
        else:
            merged.append(buf)
            buf = block

    if buf:
        merged.append(buf)

    # First block at top of resume with no recognized heading = contact info
    if merged and merged[0].section == ResumeSection.UNKNOWN:
        merged[0] = SectionBlock(
            section    = ResumeSection.CONTACT,
            heading    = merged[0].heading or "Contact",
            content    = merged[0].content,
            confidence = "inferred",
        )

    return merged


def sections_to_prompt_text(blocks: list[SectionBlock]) -> str:
    """
    Convert classified section blocks into a structured string for the LLM.
    Much cleaner input than raw resume text — LLM sees labeled sections.

    Example output:
        [CONTACT]
        Abhilash K S | abhilash@email.com | +91-...

        [SKILLS]
        Python, FastAPI, Next.js, PostgreSQL, Docker
        ...
    """
    parts = []
    for block in blocks:
        label = block.section.value.upper().replace("_", " ")
        parts.append(f"[{label}]\n{block.content.strip()}")
    return "\n\n".join(parts)