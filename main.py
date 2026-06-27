import asyncio
from Reader.DocReader import extract_resume_text
from Services.candidateResumeLLMService import aiResumeParser
from utils.section_classifier import classify_sections, sections_to_prompt_text
from rich import print

async def main():
    text = extract_resume_text("Abhilash_KS_Resume.pdf")

    # See exactly what sections were detected (great for debugging)
    blocks = classify_sections(text)
    print("\n[bold cyan]Detected Sections:[/bold cyan]")
    for b in blocks:
        print(f"  [{b.confidence}] {b.section.value.upper()} — {len(b.content)} chars")

    result = await aiResumeParser(text)
    candidate = result.model_dump(mode="json")
    print("\n[bold green]Parsed Result:[/bold green]", candidate)

asyncio.run(main())