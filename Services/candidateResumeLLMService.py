from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from Dtos.CandidateDTOs import CandidateProfile
from utils.section_classifier import classify_sections, sections_to_prompt_text
import os

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-large-latest",
    temperature=0
)

SYSTEM_PROMPT = """You are an expert resume parser for an ATS system.
The resume text below has been pre-labeled with section tags like [CONTACT], [SKILLS], [WORK EXPERIENCE], etc.
Use these section labels to accurately extract and structure the candidate's information.
Be precise. If a field is not present, return null. Do not infer or hallucinate values."""


async def aiResumeParser(text: str) -> CandidateProfile:
    # Classify sections first
    blocks = classify_sections(text)
    structured_text = sections_to_prompt_text(blocks)

    # Now LLM receives labeled sections, not raw blob
    structured_llm = llm.with_structured_output(CandidateProfile)
    result = await structured_llm.ainvoke(
        f"{SYSTEM_PROMPT}\n\nRESUME:\n{structured_text}"
    )
    return result