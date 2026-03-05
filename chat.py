import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ========== Setupt Vector DB and LLM ==========
# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "mps"}  # M4 GPU
)

# 1. Vector DB Load
vector_db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# 2. LLM Load
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOllama(model="llama3.1",
                 base_url = "http://127.0.0.1:11435",
                 temperature=0) # For accurate answers
# To start ollama server, run: OLLAMA_HOST=127.0.0.1:11435   ollama serve

# ========== Prompt Template ==========
CUSTOM_PROMPT_TEMPLATE = """You are a K-Pop entertainment expert in financial and business analysis.
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

Reference Information:
{context}

Question: {question}

Provide your answer with confidence score:"""

custom_prompt =  PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# 3. reate RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),  # Top 5 documents
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True  # To return source documents along with the answer
)

# ========== Similarity Score Calculator ==========
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
        print(f"주의: 유사도 계산 오류: {e}")
    
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

# ========== Chat Function ==========
def chat_with_ollama(user_question: str):
    """
    Chat with Ollama LLM using vector DB retrieval and similarity scoring.
    
    Returns:
        dict: Contains 'answer', 'avg_score', 'confidence_level', 'sources'
    """
    try:
        print(f"\n검색 중: '{user_question}'")
        
        # 1. Calculate similarity score from vector DB
        avg_score, confidence_level, similarity_scores, has_low_quality = calculate_similarity_score(user_question, top_k=5)
        
        # 2. Add warning if search quality is poor
        if has_low_quality:
            print(f"경고: 검색 결과의 관련성이 낮습니다.")
            print(f"   The answer may be inaccurate or incomplete.")
        
        # 2. Get answer from QA chain
        result = qa_chain.invoke({"query": user_question})
        
        # 3. Extract answer and confidence level from LLM response
        answer_text = result['result']
        
        # Extract confidence score from LLM response if provided
        llm_confidence = extract_confidence_from_answer(answer_text)
        
        # 4. Adjust final confidence: Use vector similarity if LLM confidence is unclear
        final_confidence = min(llm_confidence / 10, avg_score) if llm_confidence else avg_score
        
        # Determine final confidence level
        if final_confidence >= 0.70:
            final_confidence_level = "Very High"
        elif final_confidence >= 0.55:
            final_confidence_level = "High"
        elif final_confidence >= 0.40:
            final_confidence_level = "Medium"
        elif final_confidence >= 0.25:
            final_confidence_level = "Low"
        else:
            final_confidence_level = "Very Low"
        
        # 3. Print answer
        print(f"\n답변:\n{answer_text}")
        
        # 4. Print accuracy score
        print(f"\n정확도 지표:")
        print(f"   * 벡터 유사도 점수: {avg_score:.1%}")
        print(f"   * 신뢰도: {final_confidence_level}")
        print(f"   * 참고 문서 수: {len(result.get('source_documents', []))}")
        
        # 5. Print quality warning if needed
        if final_confidence < 0.40:
            print(f"\n   주의: 이 답변의 신뢰도가 낮습니다.")
            print(f"      Please verify with additional sources or ask for more specific information.")
        
        # 6. Print individual similarity scores
        if similarity_scores:
            print(f"\n   Document Relevance Scores:")
            for i, score in enumerate(similarity_scores, 1):
                print(f"     Doc {i}: {score:.1%}")
        
        # 7. Print reference information
        if result.get('source_documents'):
            print(f"\n참고 자료 ({len(result['source_documents'])}개 출처):")
            for i, doc in enumerate(result['source_documents'], 1):
                source_info = doc.metadata.get('source', 'Unknown')
                company = doc.metadata.get('company', '?')
                year = doc.metadata.get('year', '?')
                quarter = doc.metadata.get('quarter', '?')
                print(f"   {i}. {company} {year} {quarter} - {source_info}")
        
        # 8. Return structured result
        return {
            'answer': answer_text,
            'avg_score': avg_score,
            'confidence_level': final_confidence_level,
            'similarity_scores': similarity_scores,
            'sources': [doc.metadata for doc in result.get('source_documents', [])],
            'num_sources': len(result.get('source_documents', [])),
            'has_low_quality': has_low_quality
        }
        
    except Exception as e:
        print(f"\n오류: {e}")
        print("   Please check if Ollama server is running:")
        print("   OLLAMA_HOST=127.0.0.1:11435 ollama serve")
        return None


def extract_confidence_from_answer(answer_text: str) -> int:
    """
    Extract confidence score from LLM answer text.
    Looks for patterns like "Confidence: 7" or "confidence score: 8"
    
    Returns:
        int: Confidence score (1-10), or None if not found
    """
    import re
    
    # Look for confidence patterns
    patterns = [
        r'confidence[:\s]+(\d+)',
        r'confidence\s+score[:\s]+(\d+)',
        r'score[:\s]+(\d+)',
        r'\*\*confidence[:\s]+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_text.lower())
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    
    return None

# ==================== Chat Function ====================
def main():
    print("=" * 60)
    print("K-POP Chat Bot - Vector DB based")
    print("=" * 60)
    print("'quit' or 'exit'to end chat.\n")
    
    while True:
        try:
            user_input = input("Enter you questions: ").strip()
            
            if not user_input:
                print("Enter your questions.")
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nEnd Chat")
                break
            
            # Process user input
            chat_with_ollama(user_input)
            
        except KeyboardInterrupt:
            print("\n\n End Chat")
            break


if __name__ == "__main__":
    main()