import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = GoogleGenerativeAI(
    
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([
    """"
    Ans the question using only the context 
    
    context : {context}
    question : {question}
    
    
    """
])

chain = {
    
    "context": lambda x: "Python is a high level language",
    "question": RunnablePassthrough()
    
} | prompt | model

response = chain.invoke("what is python?")

print(response)
