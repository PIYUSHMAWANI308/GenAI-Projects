import os
import tempfile
import chromadb
import streamlit as st  
from dotenv import load_dotenv
from langchain_classic.chains import (create_history_aware_retriever,create_retrieval_chain)
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import (ChatMessageHistory)
from langchain_community.document_loaders import (PyPDFLoader)
from langchain_community.embeddings import (HuggingFaceEmbeddings)
from langchain_core.chat_history import (BaseChatMessageHistory)
from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder)
from langchain_core.runnables.history import (RunnableWithMessageHistory)
from langchain_groq import ChatGroq
from langchain_text_splitters import (RecursiveCharacterTextSplitter)

# Set up Streamlit
st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚"
)

st.title("📚 Conversational RAG with PDF")
st.write("Upload PDFs and chat with their content.")

# Session state
if "store" not in st.session_state:
    st.session_state.store = {}

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Input Groq API key
api_key = st.text_input(
    "Enter your Groq API Key:",
    type="password"
)

# Check if API key is provided
if not api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

# Set up Groq LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0
)

# Set up HuggingFace embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

embedding = get_embeddings()

# Session ID
session_id = st.text_input(
    "Session ID",
    value="default_session"
)

# Upload PDF files
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

        finally:
            os.remove(temp_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    # Create Chroma vector database
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    try:
        client.delete_collection("pdf_documents")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=client,
        collection_name="pdf_documents"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Create history-aware retriever
    contextualize_q_system_prompt = """
Given a chat history and the latest user question,
formulate a standalone question that can be understood
without the chat history.
Do not answer the question.
Only reformulate it when necessary.
"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # Create question-answering chain
    system_prompt = """
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer from the context, say you don't know.
Keep the answer concise and accurate.
Context:
{context}
"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    # Create RAG chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    st.session_state.rag_chain = rag_chain
    st.success(
        f"Processed {len(uploaded_files)} PDF(s) successfully!"
    )

# Manage chat history
def get_session_history(
    session: str
) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create conversational RAG chain
if st.session_state.rag_chain:
    conversational_rag_chain = RunnableWithMessageHistory(
        st.session_state.rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Ask questions
    user_input = st.text_input(
        "Ask a question about your PDF:"
    )

    if user_input:

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        # Display answer
        st.subheader("🤖 Assistant")
        st.write(response["answer"])

        # Display chat history
        st.subheader("💬 Chat History")

        history = get_session_history(session_id)

        for message in history.messages:

            if message.type == "human":
                st.markdown(
                    f"**You:** {message.content}"
                )

            elif message.type == "ai":
                st.markdown(
                    f"**Assistant:** {message.content}"
                )
