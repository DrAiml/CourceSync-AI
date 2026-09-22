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


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

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
# CREATE RAG PIPELINE
# =========================================================

@st.cache_resource
def create_rag_pipeline(pdf_path):

    # -----------------------------------------------------
    # 1. LOAD PDF
    # -----------------------------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # -----------------------------------------------------
    # 2. SPLIT DOCUMENT INTO CHUNKS
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)


    # -----------------------------------------------------
    # 3. CREATE EMBEDDINGS
    # -----------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


    # -----------------------------------------------------
    # 4. CREATE VECTOR STORE
    # -----------------------------------------------------

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="streamlit_pdf_rag"
    )


    # -----------------------------------------------------
    # 5. CREATE RETRIEVER
    # -----------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


    # -----------------------------------------------------
    # 6. CREATE GEMINI MODEL
    # -----------------------------------------------------

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )


    # -----------------------------------------------------
    # 7. CREATE PROMPT
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
    # 8. CREATE RAG CHAIN
    # -----------------------------------------------------

    rag_chain = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt | model


    # Return pipeline + document information
    return rag_chain, len(documents), len(chunks)


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

    # -----------------------------------------------------
    # SAVE UPLOADED PDF
    # -----------------------------------------------------

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())


    st.success("PDF uploaded successfully! ✅")


    # -----------------------------------------------------
    # CREATE RAG PIPELINE
    # -----------------------------------------------------

    with st.spinner("Processing PDF..."):

        rag_chain, pages, chunks = create_rag_pipeline(
            "uploaded.pdf"
        )


    # -----------------------------------------------------
    # SHOW PDF INFORMATION
    # -----------------------------------------------------

    st.write(f"📄 Pages: {pages}")
    st.write(f"✂️ Chunks: {chunks}")


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    if question:

        with st.spinner("Thinking..."):

            response = rag_chain.invoke(question)


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        st.subheader("💬 Answer")


        # Handle Gemini's structured content response
        if isinstance(response.content, str):

            answer = response.content

        else:

            answer = "".join(
                block["text"]
                for block in response.content
                if isinstance(block, dict)
                and block.get("type") == "text"
            )


        st.write(answer)