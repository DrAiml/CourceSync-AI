from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

#embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2"
)

#creating text
texts =[
    "Python is a high-level programming language.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning works with unlabeled data.",
    "Deep learning uses neural networks."
]

#create vector store
vector_store = Chroma.from_texts(
    texts = texts,
    embedding= embeddings,
    collection_name="retriever_demo"
)

# retriever
retriever = vector_store.as_retriever(
    kwargs={"k":2}
) 

# ask question 
question = "what is supervised learning"

# Retrieve relavant document
document = retriever.invoke(question)

# print results
print("\nRealavant Ducuments\n")

for doc in document:
    print(doc.page_content)
