# K-Pop Business Analysis Chat Bot

AI-powered chatbot for analyzing K-POP industry financial and business data using Vector DB and local LLM.

**Language:** [English](README.md) | [한글 (Korean)](README_KO.md)

## Overview

This chatbot leverages Ollama's local LLM (Llama 3.1) and ChromaDB vector database to analyze K-POP industry financial data and business information. Unlike prompt-based approaches, it uses **Vector DB similarity scores** to objectively evaluate answer accuracy.

## Features

- Local LLM-based: Ollama + Llama 3.1 with no cloud dependency
- Vector DB Search: ChromaDB for efficient document retrieval
- Accuracy Scoring: Automatic reliability calculation based on similarity (0-100%)
- Multiple Confidence Levels: Very High / High / Medium / Low / Very Low
- Streamlit Web UI: Intuitive and user-friendly interface
- Chat History: Automatic conversation logging
- Source References: Document citations for all answers

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama + Llama 3.1 |
| Embedding | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Web UI | Streamlit |
| Language | Python 3.9+ |

## Requirements

- Python 3.9+
- Ollama (Install: https://ollama.ai)
- 4GB+ RAM

## Installation & Usage

### 1. Activate Virtual Environment
```bash
source kpopBot_venv/bin/activate
```

### 2. Start Ollama Server
```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

**Install required model:**
```bash
ollama pull llama3.1
```

### 3. Run Web UI (Recommended)
```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

### Or use CLI
```bash
python chat.py
```

## Accuracy Scoring System

Automatically calculated based on Vector DB Cosine Similarity:

| Score | Confidence | Description |
|-------|-----------|-------------|
| 85%+ | Very High | Highly trustworthy |
| 70-85% | High | Reliable |
| 55-70% | Medium | Moderately reliable |
| 40-55% | Low | Low confidence |
| <40% | Very Low | Not trustworthy |

## Project Structure

```
KpopBot/
├── app.py                 # Streamlit web UI
├── chat.py               # CLI chatbot
├── fileProcess.py        # PDF processing and vector DB creation
├── Data/                 # K-POP company financial documents (PDF)
│   ├── 2021년/
│   ├── 2022년/
│   ├── 2023년/
│   ├── 2024년/
│   └── 2025년/
├── vector_db/            # ChromaDB storage
└── README.md
```

## File Descriptions

### `app.py`
- Streamlit-based web interface
- Visual accuracy metrics display
- Detailed analysis view (per-document similarity)
- Chat history management

### `chat.py`
- Terminal-based CLI interface
- Text-based accuracy score output
- Automatic Q&A loop

### `fileProcess.py`
- PDF document loading and preprocessing
- Text chunking
- Vector embedding and ChromaDB storage
- Metadata management (year, quarter, company)

## Usage Examples

**Sample Questions:**
- "What was HYBE's revenue in 2024?"
- "What is SM Entertainment's quarterly operating profit trend?"
- "What are JYP Entertainment's main revenue sources?"

**Response Includes:**
1. Detailed answer in markdown format
2. Accuracy metrics
3. Per-document similarity scores
4. Source references

## Configuration

Modify settings in `chat.py` or `app.py`:

```python
# LLM settings
llm = ChatOllama(
    model="llama3.1",
    base_url="http://127.0.0.1:11435",  # Ollama address
    temperature=0  # 0: accurate, 1: creative
)

# Search options
retriever=vector_db.as_retriever(search_kwargs={"k": 5})  # Top 5 documents
```

## Troubleshooting

**Q: "Connection refused" error**
```
Ollama server not started
$ OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

**Q: "Vector DB not found"**
```
vector_db folder missing
$ python fileProcess.py  # Process data to create vector DB
```

**Q: Low accuracy score**
```
Need more training data
- Add more PDFs to Data/ folder
- Run python fileProcess.py to re-index
```

## License

MIT License

---

**Last Updated**: March 2026
