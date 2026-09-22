import streamlit as st

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

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions from it.")


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully! ✅")


    # -----------------------------------------------------
    # LOAD PDF
    # -----------------------------------------------------

    loader = PyPDFLoader("uploaded.pdf")

    documents = loader.load()

    st.write(f"📄 Pages: {len(documents)}")


    # -----------------------------------------------------
    # SPLIT INTO CHUNKS
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    st.write(f"✂️ Chunks: {len(chunks)}")


    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


    # -----------------------------------------------------
    # VECTOR STORE
    # -----------------------------------------------------

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="streamlit_pdf_rag"
    )


    # -----------------------------------------------------
    # RETRIEVER
    # -----------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer the question using ONLY the provided context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the document,
    say:

    "I don't know based on the provided document."
    """)


    # -----------------------------------------------------
    # RAG CHAIN
    # -----------------------------------------------------

    rag_chain = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt | model


    # =====================================================
    # QUESTION
    # =====================================================

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    if question:

        with st.spinner("Thinking..."):

            response = rag_chain.invoke(question)

        st.subheader("💬 Answer")

        st.write(response.content[0]["text"])