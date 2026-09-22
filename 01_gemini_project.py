from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Gemini api key is missing")

llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    temprature=0.7
    )

response = llm.invoke("Say 'Setup complete and working fine!' in 1 sentence.")

print(response)