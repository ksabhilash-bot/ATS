import asyncio

from Reader.JobDescriptionReader import clean_jd_text
from Services.jobDescriptionLLMService import JobDescriptionParser

from rich import print


async def main():

    text = input("Enter a text:")
    print("*"* 50)
    clean_text=clean_jd_text(text)
    print("cleaned text:",clean_text)
    result=await JobDescriptionParser(clean_text)
    result = result.model_dump(mode="json")
    print("*" * 50)
    print(result)
    


asyncio.run(main())
