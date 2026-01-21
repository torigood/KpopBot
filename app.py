import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

# --- Set up ---
st.set_page_config(page_title="K-Pop Business Analysis Chat Bot", layout="wide")
st.title("🤖 K-Pop Business Analysis Chat Bot")
st.markdown("Vector DB and Ollama-based real-time Q&A service.")

# --- Reset Logic (Processing Cache) ---
# Use @st.cache_resource to avoid reloading the model on every page reload.
@st.cache_resource
def init_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "mps"}
    )
    
    vector_db = Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )
    
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://127.0.0.1:11435",
        temperature=0
    )

    template = """You are a K-Pop entertainment expert financial and business analysis...
    Context: {context}
    Question: {question}
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

qa_chain = init_qa_chain()

# --- Manage chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Print chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("K-Pop business questions"):
    # 1. Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Chat Bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result['result']
                
                # Print bot response in chat message container
                st.markdown(response)
                
                # Reference documents
                if result.get('source_documents'):
                    with st.expander("📚 References"):
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"**References {i}:** {doc.metadata.get('source', 'Unknown')}")
                            st.caption(doc.page_content[:200] + "...")

                # Save bot response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error occurred: {e}")