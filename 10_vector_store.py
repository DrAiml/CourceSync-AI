from dotenv import  load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
      model = "gemini-embedding-2"
)

texts = [
     "Python is a high-level programming language.",
     "Supervised learning uses labeled training data.",
     "Unsupervised learning works with unlabeled data.",
     "Deep learning uses neural networks."
]

vector_store = Chroma.from_texts(
    texts = texts,
    embedding= embeddings,
    collection_name="learning_demo"
)

result = vector_store.similarity_search(
    "what is similarity search?",
    k=2
)

print("\nRelavant Ducument")

for doc in result:
    print(doc.page_content)
    