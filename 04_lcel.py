import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(
    
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([
    
    ("system", "you are expert in {space}"),
    ("human", "explain me {topic} in 50 words")
    
])

chain = prompt | model

response = chain.invoke({
    
    "space":"Medicine",
    "topic":"albendazol"
})

print(response)