from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# load pdf
loader = PyPDFLoader("07_notes.pdf")

documents = loader.load()

print("total pages:", len(documents))

# split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(documents)

print("total chunks:", len(chunks))

# embedding model 

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

# vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="pdf_rag_demo"
) 

# create retiever
retriever = vector_store.as_retriever(
    search_kwargs ={"k":3}
)

# gemini model
model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    temperature=0
)

# prompt 
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say: "I don't know based on the provided document."
"""
)

# rag chain 
rag_chain = {
    "context": retriever,
    "question": RunnablePassthrough()
} | prompt | model

# question 
question = "What are transition elements?"

docs = retriever.invoke(question)

print("\n========== RETRIEVED CHUNKS ==========")

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc.page_content)
    print("\nMetadata:", doc.metadata)
    
# run rag 
responses = rag_chain.invoke(question)   

# print answer 
print("\nQuestion:")
print(question)

print("\nAnswer:")
print(responses.content[0]["text"])