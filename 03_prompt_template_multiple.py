import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(
    
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([
    
    ("system", "You are an expert in {space}"),
    ("human", "Explain me {topic} in {style} in {words}")
    
])

prompt_value = prompt.invoke({
    
    "space": "chemicals",
    "topic": "wurtz reaction",
    "style": "begginer",
    "words": "50"
})

response = model.invoke(prompt_value)
print(response)