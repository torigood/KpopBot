# 🤖 K-Pop Business Analysis Chat Bot

**벡터 DB 기반의 K-POP 비즈니스 분석 AI 챗봇**

## 개요

Ollama의 로컬 LLM(Llama 3.1)과 ChromaDB 벡터 데이터베이스를 활용하여 K-POP 산업 관련 재무 및 비즈니스 데이터를 분석하는 챗봇입니다. 프롬프트 기반이 아닌 **벡터 DB 유사도 점수**로 답변의 정확도를 객관적으로 평가합니다.

## ✨ 주요 기능

- ✅ **로컬 LLM 기반**: Ollama + Llama 3.1 사용으로 개인정보 보호
- ✅ **벡터 DB 검색**: ChromaDB를 통한 효율적인 문서 검색
- ✅ **정확도 점수**: 유사도 기반 신뢰도 자동 계산 (0~100%)
- ✅ **다양한 신뢰도 레벨**: Very High / High / Medium / Low / Very Low
- ✅ **Streamlit UI**: 시각적이고 사용자 친화적인 웹 인터페이스
- ✅ **채팅 히스토리**: 대화 기록 자동 저장
- ✅ **참고 자료**: 답변 기반의 출처 문서 제공

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|---------|------|
| **LLM** | Ollama + Llama 3.1 |
| **Embedding** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Web UI** | Streamlit |
| **언어** | Python 3.9+ |

## 📋 사전 요구사항

- Python 3.9+
- Ollama (설치: https://ollama.ai)
- 4GB+ RAM

## 🚀 설치 및 실행

### 1️⃣ 가상환경 활성화
```bash
source kpopBot_venv/bin/activate
```

### 2️⃣ Ollama 서버 시작
```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

**필요한 모델 설치:**
```bash
ollama pull llama3.1
```

### 3️⃣ 웹 UI 실행 (추천)
```bash
streamlit run app.py
```

🌐 자동으로 브라우저에서 `http://localhost:8501` 열림

### 또는 CLI 사용
```bash
python chat.py
```

## 📊 정확도 점수 시스템

벡터 DB의 유사도(Cosine Similarity)를 기반으로 자동 계산:

| 점수 | 신뢰도 | 설명 |
|------|--------|------|
| 85% 이상 | Very High ✅ | 매우 신뢰할 수 있음 |
| 70~85% | High ✅ | 신뢰할 수 있음 |
| 55~70% | Medium ⚠️ | 상대적으로 신뢰 |
| 40~55% | Low ⚠️ | 낮은 신뢰도 |
| 40% 미만 | Very Low ❌ | 신뢰할 수 없음 |

## 📁 프로젝트 구조

```
KpopBot/
├── app.py                 # Streamlit 웹 UI
├── chat.py               # CLI 챗봇
├── fileProcess.py        # PDF 데이터 처리 및 벡터 DB 생성
├── Data/                 # K-POP 기업 재무 자료 (PDF)
│   ├── 2021년/
│   ├── 2022년/
│   ├── 2023년/
│   ├── 2024년/
│   └── 2025년/
├── vector_db/            # ChromaDB 저장소
└── README.md
```

## 🔧 주요 파일 설명

### `app.py`
- Streamlit 기반 웹 인터페이스
- 시각적 정확도 메트릭 표시
- 상세 분석 보기 (문서별 유사도)
- 채팅 히스토리 관리

### `chat.py`
- 터미널 기반 CLI 인터페이스
- 텍스트 기반 정확도 점수 출력
- 자동 질문-응답 루프

### `fileProcess.py`
- PDF 문서 로딩 및 전처리
- 텍스트 분할 (Chunking)
- 벡터 임베딩 및 ChromaDB 저장
- 메타데이터 관리 (연도, 분기, 회사명)

## 💬 사용 예시

**질문 예시:**
- "HYBE의 2024년 매출은?"
- "SM엔터테인먼트의 분기별 영업이익 추이는?"
- "JYP 엔터테인먼트의 주요 수익원은?"

**응답 구성:**
1. 상세한 답변 (마크다운 형식)
2. 📊 정확도 메트릭
3. 📈 문서별 유사도 점수
4. 📚 참고 자료 (출처)

## ⚙️ 설정

`chat.py` 또는 `app.py`의 설정 변경:

```python
# LLM 설정
llm = ChatOllama(
    model="llama3.1",
    base_url="http://127.0.0.1:11435",  # Ollama 주소
    temperature=0  # 0: 정확한 답변, 1: 창의적 답변
)

# 검색 옵션
retriever=vector_db.as_retriever(search_kwargs={"k": 5})  # 상위 5개 문서
```

## 🐛 문제 해결

**Q: "Connection refused" 에러**
```
⚠️ Ollama 서버를 시작하지 않음
$ OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

**Q: "Vector DB not found"**
```
⚠️ vector_db 폴더가 없음
$ python fileProcess.py  # 데이터 처리하여 벡터 DB 생성
```

**Q: 낮은 정확도 점수**
```
💡 더 많은 학습 데이터 추가 필요
- Data/ 폴더에 더 많은 PDF 추가
- python fileProcess.py 실행하여 재색인
```

## 📝 라이선스

MIT License

## 👨‍💻 저자

K-POP Business Analysis Chat Bot Project

---

**마지막 업데이트**: 2026년 3월
