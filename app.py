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
st.markdown("Vector DB and Ollama-based real-time Q&A service with accuracy scoring.")

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

# --- Similarity Score Calculator ---
@st.cache_resource
def get_vector_db():
    """Get cached vector DB instance"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "mps"}
    )
    return Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )

vector_db = get_vector_db()

def calculate_similarity_score(user_question: str, top_k: int = 5) -> tuple:
    """
    Calculate average similarity score from vector DB search results.
    
    Returns:
        tuple: (average_score, confidence_level, similarity_scores)
    """
    # Search with scores from vector DB
    results_with_scores = vector_db.similarity_search_with_scores(user_question, k=top_k)
    
    if not results_with_scores:
        return 0.0, "Low", []
    
    # Extract scores (Chroma returns scores as distances, convert to similarity)
    # Lower distance = higher similarity, so we use 1 / (1 + distance)
    similarity_scores = [1 / (1 + score) for _, score in results_with_scores]
    avg_score = sum(similarity_scores) / len(similarity_scores)
    
    # Determine confidence level
    if avg_score >= 0.85:
        confidence_level = "Very High"
    elif avg_score >= 0.70:
        confidence_level = "High"
    elif avg_score >= 0.55:
        confidence_level = "Medium"
    elif avg_score >= 0.40:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return avg_score, confidence_level, similarity_scores

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
                # Calculate similarity score
                avg_score, confidence_level, similarity_scores = calculate_similarity_score(prompt, top_k=5)
                
                # Get QA response
                result = qa_chain.invoke({"query": prompt})
                response = result['result']
                
                # Display bot response
                st.markdown(response)
                
                # Display accuracy metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similarity Score", f"{avg_score:.1%}")
                with col2:
                    st.metric("Confidence", confidence_level)
                with col3:
                    st.metric("Sources Used", len(result.get('source_documents', [])))
                
                # Expandable details
                with st.expander("📈 Detailed Accuracy Analysis"):
                    if similarity_scores:
                        st.write("**Individual Document Similarity Scores:**")
                        for i, score in enumerate(similarity_scores, 1):
                            progress_value = min(score, 1.0)  # Ensure value is between 0-1
                            st.progress(progress_value, text=f"Document {i}: {score:.1%}")
                    
                    # Add warning for low scores
                    if avg_score < 0.4:
                        st.warning("**Low Relevance Warning**: The search results have low similarity to your question. The answer may not be accurate.")
                    elif avg_score < 0.55:
                        st.info("**Moderate Relevance**: Consider verifying the answer with additional sources.")
                
                # Reference documents
                if result.get('source_documents'):
                    with st.expander("References"):
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"**Reference {i}:** {doc.metadata.get('source', 'Unknown')}")
                            st.caption(doc.page_content[:200] + "...")

                # Save bot response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error occurred: {e}")