from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from Dtos.CandidateDTOs import CandidateProfile
import os
load_dotenv()

MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-large-latest",
    temperature=0
)


async def aiResumeParser(text:str)->CandidateProfile:
    structured_llm = llm.with_structured_output(CandidateProfile)
    result=await structured_llm.ainvoke(text)
    return result