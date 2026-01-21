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

# ========== Chat Function ==========
def chat_with_ollama(user_question: str):
    """
    result = qa_chain.invoke({"query": user_question})
    
    # Calculate average score from source documents
    avg_score = sum([doc.metadata.get('score', 0) 
                     for doc in result['source_documents']]) / len(result['source_documents'])
    
    confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.5 else "Low"
    print(f"Confidence: {confidence} (Score: {avg_score:.2f})")
    
    if avg_score < 0.5:
        print(" Warning: Not enought Documents.")
    """
    try:
        print(f"\n Searching...: '{user_question}'")
        result = qa_chain.invoke({"query": user_question})
        
        # Print answer
        print(f"\n Answers \n{result['result']}")
        
        # Print reference information (optional)
        if result.get('source_documents'):
            print(f"\n Referecne Info ({len(result['source_documents'])}개):")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"  {i}. {doc.metadata}")
        
        return result['result']
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("    Ollama: Check Ollama server:")
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