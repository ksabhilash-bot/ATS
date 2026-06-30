import asyncio
from Reader.DocReader import extract_resume_text
from Services.candidateResumeLLMService import aiResumeParser
from utils.hybrid_section_classifier import classify_sections_hybrid
from rich import print

async def main():
    text = extract_resume_text("Abhilash_KS_Resume.pdf")

    # See exactly what sections were detected and which tier produced each one
    blocks = await classify_sections_hybrid(text)
    print("\n[bold cyan]Detected Sections:[/bold cyan]")
    for b in blocks:
        print(f"  {b.confidence:>10s} | {b.section.value.upper()} — {len(b.content)} chars")

    result = await aiResumeParser(text)
    candidate = result.model_dump(mode="json")
    print("\n[bold green]Parsed Result:[/bold green]", candidate)

asyncio.run(main())