from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
from Dtos.JobDescriptionDTOs import JobDescription
load_dotenv()
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")


llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small-2603",
    temperature=0
)

async def JobDescriptionParser(text:str)->JobDescription:
    structured_llm = llm.with_structured_output(JobDescription)
    result =await structured_llm.ainvoke(text)
    return result
