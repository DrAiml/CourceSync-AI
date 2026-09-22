from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("07_notes.pdf")

documents = loader.load()

print("total pages:", len(documents))

for i, doc in enumerate(documents[:3]):
    print(f"/n--- PAGE {i + 1} ---")
    print(doc.page_content[:500])
    print("Metadata: ", doc.metadata)
    
    