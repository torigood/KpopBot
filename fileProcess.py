import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def process_pdf(folder_path):
    all_chuncks = []
    spliter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 100)

    # Localliy embedding model
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "mps"}  # M4 GPU
)
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Skip hidden files
            if file_name.startswith('.'):
                continue
                
            if file_name.endswith('.pdf'):
                # 1. Split the file name
                # EX) "HYBE 2021 Q1.pdf" -> ["HYBE", "2021", "Q1"]
                name_parts = file_name.replace(".pdf", "").split(" ")
                
                # 2. Extract company name, year, and quarter
                company = name_parts[0] if len(name_parts) > 0 else "Unknown"
                year = name_parts[1] if len(name_parts) > 1 else "Unknown"
                quarter = name_parts[2] if len(name_parts) > 2 else "Unknown"
            
                # 3. PDF Loading
                try:
                    loader = PyPDFLoader(os.path.join(root, file_name))
                    pages = loader.load()

                    for page in pages:
                        page.metadata = {
                            "company": company.upper(),
                            "year": int(year) if year.isdigit() else 0,
                            "quarter": quarter.upper(),
                            "source": file_name
                        }

                    # 4. Text Splitting
                    chunks = spliter.split_documents(pages)
                    all_chuncks.extend(chunks)
                    print("[INFO] Processed file:", file_name, "\n")
                    print("[INFO] Number of chunks created:", len(chunks), "\n")
                    print(f"Completed: {company} | {year} | {quarter}\n")
                except Exception as e:
                    print(f"Error Failed to process file: {file_name}: {e}")

    # 5. Save to Vector DB
    if all_chuncks:  # Save if chunck exist
        print(f"\n[INFO] Saving total {len(all_chuncks)} vectors to DB")
        vector_db = Chroma.from_documents(
            documents = all_chuncks,
            embedding=embeddings,
            persist_directory="./vector_db"
        )
    else:
        print("[WARNING] No chunks to save to vector DB")
        vector_db = None

    return vector_db

data_folder = "./Data"
v_db = process_pdf(data_folder)
if v_db:
    print("\nSuccess. Vector DB is ready to use.")
else:
    print("\nFailed. No data was processed.")