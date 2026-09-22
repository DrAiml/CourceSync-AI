import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(
    
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([

 ("system", "You are a helpful AI tutor"),
 ("human", "Explain {topic} in simple language")
 
 ])

prompt_value = prompt.invoke({
    
    "topic": "machine learning"
})

response = model.invoke(prompt_value)
print(response)