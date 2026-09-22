from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# loading the pdf
loader = PyPDFLoader("07_notes.pdf")
documents = loader.load()

# creating text splitter
splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 1000,
    chunk_overlap = 200
)

# splitting documents into chunks
chunks = splitter.split_documents(documents)

print("total chunks", len(chunks))

# see first chunk

print("\n--- First chunk ---")
print(chunks[0].page_content)

# see metadata

print("\n--- Metadata---")
print(chunks[0].metadata)

