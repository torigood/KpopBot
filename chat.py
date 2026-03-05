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
CUSTOM_PROMPT_TEMPLATE = """You are a K-Pop entertainment expert financial and business analysis.
Based on the given information, answer the questions as accurately as possible.
- If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
- Provide your answer in markdown format.
- Provide references to support your answer using the source documents provided(Its name).
- Use bullet points, tables, and other formatting tools to make the answer easy to read and understand.
- Provide concise and clear answers.
- If there is specific data or statistics, cite them properly.
- If possible, provide year and quarter for the data.

- At the end of your answer, provide a summary of key points in bullet format.
- At the end of you answer, provide your answers trusted scores based on the following criteria:
  - 0-2: Low confidence, information may be inaccurate or incomplete.
  - 3-5: Moderate confidence, information is somewhat reliable but may lack depth.
  - 6-8: High confidence, information is reliable and well-supported.
  - 9-10: Very high confidence, information is accurate, comprehensive, and well-supported.

Reference Information:
{context}

Question: {question}
Answer: """

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

# ========== Chat Function ==========
def chat_with_ollama(user_question: str):
    """
    Chat with Ollama LLM using vector DB retrieval and similarity scoring.
    
    Returns:
        dict: Contains 'answer', 'avg_score', 'confidence_level', 'sources'
    """
    try:
        print(f"\n Searching...: '{user_question}'")
        
        # 1. Calculate similarity score from vector DB
        avg_score, confidence_level, similarity_scores = calculate_similarity_score(user_question, top_k=5)
        
        # 2. Get answer from QA chain
        result = qa_chain.invoke({"query": user_question})
        
        # 3. Print answer
        print(f"\n Answer:\n{result['result']}")
        
        # 4. Print accuracy score based on vector DB similarity
        print(f"\n Accuracy Score:")
        print(f"   Average Similarity: {avg_score:.2%}")
        print(f"   Confidence Level: {confidence_level}")
        
        # Determine recommendation based on score
        if avg_score < 0.4:
            print(f"    Warning: Low relevance. Search results may not match the question well.")
        
        # 5. Print individual similarity scores
        if similarity_scores:
            print(f"\n   Individual Scores:")
            for i, score in enumerate(similarity_scores, 1):
                print(f"     Document {i}: {score:.2%}")
        
        # 6. Print reference information
        if result.get('source_documents'):
            print(f"\n Reference Info ({len(result['source_documents'])}개):")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"   {i}. {doc.metadata}")
        
        # 7. Return structured result
        return {
            'answer': result['result'],
            'avg_score': avg_score,
            'confidence_level': confidence_level,
            'similarity_scores': similarity_scores,
            'sources': [doc.metadata for doc in result.get('source_documents', [])],
            'num_sources': len(result.get('source_documents', []))
        }
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("   Ollama: Check Ollama server:")
        print("   OLLAMA_HOST=127.0.0.1:11435 ollama serve")
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