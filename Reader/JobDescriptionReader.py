import re
BULLET_SYMBOLS = [
    "•", "●", "▪", "◦",
    "►", "▶", "➤", "✓",
    "✔", "", "■", "□"
]

def clean_jd_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove bullet symbols
    for symbol in BULLET_SYMBOLS:
        text = text.replace(symbol, " ")

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove extra newlines
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()