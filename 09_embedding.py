from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2" 
)

text = "supervise learning uses labeled data"

vector = embedding.embed_query(text)

print("original text")
print(text)

print("\nVector Length:")
print(len(vector))

print("\nFirts 10 values")
print(vector[:10])


