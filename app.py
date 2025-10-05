import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Environment Setup ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# --- Core Functions ---

def process_pdfs(uploaded_files, embeddings):
    all_documents = []
    for uploaded_file in uploaded_files:
        temp_filepath = f"./temp_{uploaded_file.name}"
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(temp_filepath)
        all_documents.extend(loader.load())
        os.remove(temp_filepath)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

def get_conversational_rag_chain(retriever, llm):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise and use a maximum of three sentences."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --- Streamlit App UI ---

st.set_page_config(page_title="Chat with PDFs")
st.title("Conversational RAG with Chat History")
st.write("Upload your PDF documents and ask questions about their content.")
# Input for Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# Check if keys are available before proceeding
if groq_api_key and hf_token:
    # Initialize the LLM and embeddings model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Choose your PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        # Using st.cache_resource to avoid reprocessing files on every interaction
        @st.cache_resource
        def create_retriever(_uploaded_files):
            with st.spinner("Processing PDFs..."):
                return process_pdfs(_uploaded_files, embeddings)
        
        retriever = create_retriever(uploaded_files)
        
        rag_chain = get_conversational_rag_chain(retriever, llm)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        session_id = "user_session"
        
        history = get_session_history(session_id)
        for msg in history.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_question := st.chat_input("Ask a question about your documents..."):
            st.chat_message("human").write(user_question)
            
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}}
                )
            
            st.chat_message("ai").write(response['answer'])
    else:
        st.info("Please upload one or more PDF files to get started.")
else:
    st.warning("Please provide your Groq API key and Hugging Face Token to proceed.")
    st.info("You can add them to a .env file locally or as Secrets when deploying.")