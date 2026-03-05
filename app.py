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

    template = """You are a K-Pop entertainment expert in financial and business analysis.
Your task is to answer questions based ONLY on the provided reference information.

**CRITICAL RULES:**
1. Base your answer ONLY on the provided context documents.
2. If the context does NOT contain sufficient information to answer the question, MUST state: "I don't have sufficient information to answer this question accurately."
3. If the question asks about specific artists/groups not mentioned in context, say you lack that specific information.
4. DO NOT make assumptions or provide general knowledge not in the context.
5. Always cite the source document for each claim (company, year, quarter).

**ANSWER FORMAT:**
- Use markdown format with clear sections
- Use bullet points for easy reading
- Cite specific data with sources (e.g., "According to HYBE 2023 Q2...")
- Provide year and quarter for all data

**CONFIDENCE ASSESSMENT:**
At the end of your answer, provide a confidence score (1-10) based on:
- 8-10: Information is directly from provided source documents, well-supported
- 6-7: Information is from source documents but may require interpretation
- 4-5: Limited source information, answer may lack depth
- 1-3: Minimal or no direct information in sources, answer is somewhat speculative

If confidence is below 5, add a disclaimer: "⚠️ Note: This answer is based on limited information from available sources."

Reference Information: {context}
Question: {question}
Provide your answer with confidence score:"""
    
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
    Calculate average similarity score from vector DB search results using actual embeddings.
    
    Returns:
        tuple: (average_score, confidence_level, similarity_scores, has_low_quality_results)
    """
    avg_score = 0.3  # Default low score for no results
    similarity_scores = []
    has_low_quality_results = False
    
    try:
        # Get retrieval results with actual similarity scores
        retriever_results = vector_db.similarity_search_with_scores(user_question, k=top_k)
        
        if retriever_results:
            # Extract actual similarity scores from vector DB
            # LangChain returns (document, score) tuples where score is distance
            # Convert distance to similarity: similarity = 1 / (1 + distance)
            similarity_scores = []
            for doc, distance in retriever_results:
                # Normalize distance to similarity score (0-1 range)
                similarity = 1 / (1 + distance) if distance < 10 else max(0, 1 - distance / 10)
                similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_score = sum(similarity_scores) / len(similarity_scores)
            else:
                avg_score = 0.3
            
            # Check if results have low quality (all scores below 0.5)
            if avg_score < 0.5:
                has_low_quality_results = True
        else:
            avg_score = 0.2  # Very low when no results
            has_low_quality_results = True
    
    except Exception as e:
        # Final fallback
        avg_score = 0.2
        similarity_scores = []
        has_low_quality_results = True
    
    # Determine confidence level - stricter thresholds
    if avg_score >= 0.80:
        confidence_level = "Very High"
    elif avg_score >= 0.65:
        confidence_level = "High"
    elif avg_score >= 0.50:
        confidence_level = "Medium"
    elif avg_score >= 0.35:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return avg_score, confidence_level, similarity_scores, has_low_quality_results

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
                avg_score, confidence_level, similarity_scores, has_low_quality = calculate_similarity_score(prompt, top_k=5)
                
                # Get QA response
                result = qa_chain.invoke({"query": prompt})
                response = result['result']
                
                # Display warning if search quality is poor
                if has_low_quality:
                    st.warning(
                        "⚠️ **Low Search Relevance**: The search results have limited relevance to your question. "
                        "This answer may be inaccurate or incomplete.",
                        icon="⚠️"
                    )
                
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
                with st.expander("📊 Detailed Accuracy Analysis"):
                    if similarity_scores:
                        st.write("**Document Relevance Scores:**")
                        for i, score in enumerate(similarity_scores, 1):
                            progress_value = min(score, 1.0)
                            st.progress(progress_value, text=f"Document {i}: {score:.1%}")
                    
                    # Add detailed warnings
                    if avg_score < 0.35:
                        st.error("**Very Low Relevance**: The search results have very low similarity to your question. "
                                "The answer is likely inaccurate. Consider asking with more specific keywords.")
                    elif avg_score < 0.50:
                        st.warning("**Low Relevance**: The search results have limited similarity. "
                                  "Verify the answer with additional sources.")
                    elif avg_score < 0.65:
                        st.info("**Moderate Relevance**: Results are somewhat relevant. Check source documents for details.")
                
                # Reference documents
                if result.get('source_documents'):
                    with st.expander("📚 References (Click to expand)", expanded=False):
                        for i, doc in enumerate(result['source_documents'], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            company = doc.metadata.get('company', '?')
                            year = doc.metadata.get('year', '?')
                            quarter = doc.metadata.get('quarter', '?')
                            st.write(f"**Reference {i}:** {company} {year} {quarter}")
                            st.caption(f"📄 {source}")

                # Save bot response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error occurred: {e}")