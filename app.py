import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

# --- Set up ---
st.set_page_config(page_title="K-Pop Business Analysis Chat Bot", layout="wide")
st.title("K-Pop Business Analysis Chat Bot")
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

If confidence is below 5, add a disclaimer: "Note: This answer is based on limited information from available sources."

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
        # Chroma uses L2 distance by default (smaller = more similar)
        retriever_results = vector_db.similarity_search_with_scores(user_question, k=top_k)
        
        if retriever_results:
            # Extract actual similarity scores from vector DB
            # LangChain returns (document, score) tuples where score is distance
            similarity_scores = []
            for doc, distance in retriever_results:
                # Chroma L2 distance: convert to similarity (0-1 range)
                # Lower distance = higher similarity
                if distance <= 1.0:
                    similarity = 1 - distance
                else:
                    similarity = 1 / (1 + distance)
                similarity_scores.append(max(0, min(1, similarity)))  # Clamp to 0-1
            
            if similarity_scores:
                avg_score = sum(similarity_scores) / len(similarity_scores)
            else:
                avg_score = 0.3
            
            # Check if results have low quality (all scores below 0.4)
            if avg_score < 0.4:
                has_low_quality_results = True
        else:
            avg_score = 0.2  # Very low when no results
            has_low_quality_results = True
    
    except Exception as e:
        # Final fallback
        avg_score = 0.2
        similarity_scores = []
        has_low_quality_results = True
    
    # Determine confidence level - calibrated thresholds
    if avg_score >= 0.70:
        confidence_level = "매우 높음"
    elif avg_score >= 0.55:
        confidence_level = "높음"
    elif avg_score >= 0.40:
        confidence_level = "중간"
    elif avg_score >= 0.25:
        confidence_level = "낮음"
    else:
        confidence_level = "매우 낮음"
    
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
                        "**검색 결과 관련성 약함**: 검색 결과가 질문과의 관련성이 낮습니다. "
                        "이 답변이 부정확하거나 불완전할 수 있습니다."
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
                with st.expander("정확도 분석"):
                    if similarity_scores:
                        st.write("**문서 관련성 점수:**")
                        for i, score in enumerate(similarity_scores, 1):
                            progress_value = min(score, 1.0)
                            st.progress(progress_value, text=f"문서 {i}: {score:.1%}")
                    
                        st.error("**매우 낮은 관련성**: 검색 결과가 질문과의 유사도가 매우 낮습니다. "
                                "이 답변은 부정확할 가능성이 높습니다. 더 구체적인 키워드로 다시 질문해보세요.")
                    elif avg_score < 0.40:
                        st.warning("**낮은 관련성**: 검색 결과의 유사도가 낮습니다. "
                                  "다른 출처로 답변을 검증하세요.")
                    elif avg_score < 0.55:
                        st.info("**중간 관련성**: 검색 결과가 어느 정도 관련이 있습니다. 원본 문서를 확인하세요.")
                
                # Reference documents
                if result.get('source_documents'):
                    with st.expander("참고 자료 (클릭하여 전개)", expanded=False):
                        for i, doc in enumerate(result['source_documents'], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            company = doc.metadata.get('company', '?')
                            year = doc.metadata.get('year', '?')
                            quarter = doc.metadata.get('quarter', '?')
                            st.write(f"**Reference {i}:** {company} {year} {quarter}")
                            st.caption(f"{source}")

                # Save bot response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error occurred: {e}")