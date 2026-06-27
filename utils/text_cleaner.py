import re

BULLET_SYMBOLS = [
    "•",
    "●",
    "▪",
    "◦",
    "►",
    "▶",
    "➤",
    "✓",
    "✔",
    "",
    "■",
    "□",
]


def clean_resume_text(text: str) -> str:
    """
    Clean and normalize extracted resume text.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace bullet symbols with a standard bullet
    for bullet in BULLET_SYMBOLS:
        text = text.replace(bullet, "- ")

    # Remove non-printable characters EXCEPT newline and tab
    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)

    lines = []

    for line in text.split("\n"):
        # Collapse multiple spaces/tabs inside a line
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            line = _normalize_capitalization(line)

        lines.append(line)

    # Remove consecutive blank lines
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        is_blank = line == ""

        if is_blank and previous_blank:
            continue

        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def _normalize_capitalization(line: str) -> str:
    """
    Normalize lines that are completely uppercase.
    """

    if line.isupper() and len(line.split()) <= 6:
        return line.title()

    return line