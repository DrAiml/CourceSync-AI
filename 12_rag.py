from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# loading env file
load_dotenv()

# embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

# text
texts = [
     "Supervised learning uses labeled training data.",
    "Unsupervised learning works with unlabeled data.",
    "Deep learning uses neural networks.",
    "Python is a high-level programming language."
]

#vector store
vector_store = Chroma.from_texts(
    texts = texts,
    embedding=embeddings,
    collection_name="rag_demo"
)

# retriever
retriever = vector_store.as_retriever(
    kwargs={"k":2}
)

# model define
model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    temperature=0
)

# prompts 
prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assitant.
        Answer the qeuestion using only the provided context.
        
        Context : 
        {context}                                   
        
        Question:
        {question}
        
        if the answer is not available in the context,
        say:"I don't know based on the provided document."                                  
"""
)

# rag chain 
rag_chain ={
    "context":retriever,
    "question":RunnablePassthrough()
} | prompt | model

# user question 
question = "Who invented the telephone?"

# run rag
responses = rag_chain.invoke(question)

# print answer
print("\nQuestion:")
print(question)

print("\nAnswer:")
print(responses.content[0]["text"])