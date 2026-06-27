import asyncio

from Reader.DocReader import extract_resume_text
from Services.candidateResumeLLMService import aiResumeParser
from Config.supabase_config import supabase_client
from rich import print
from utils.text_cleaner import clean_resume_text

async def main():

    text = extract_resume_text("resumeAbhi.pdf")
    print("text:",text)
    result = await aiResumeParser(text)
    candidate = result.model_dump(mode="json")
    print("\nresult:",candidate)

    

asyncio.run(main())
