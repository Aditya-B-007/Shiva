# AGENT.md — `src/dataUpload/`

## Purpose
Handles **document ingestion** (PDF → text) and **hybrid retrieval** (text → ranked chunks). This is the RAG layer — used to inject domain context into a query before decision-making.

---

## Files

### `dataUploadAndPrompt.py`

**Class: `PdfProcessor`**
Extracts raw text from a PDF file and wraps it in a `DataUploadDTO`.

| Method | Signature | What it does |
|---|---|---|
| `_extract_text_from_pdf` | `(pdf_path: str) → str` | Reads all pages with `PyPDF2.PdfReader`, joins with page-break markers |
| `process_and_create_upload_dto` | `(pdf_path, file_name, user_prompt=None) → DataUploadDTO` | Extracts text, wraps in DTO with optional `UserPromptDTO` |

**Usage:**
```python
from src.dataUpload import PdfProcessor

proc = PdfProcessor()
dto = proc.process_and_create_upload_dto("report.pdf", "report.pdf", "What are the risks?")
print(dto.fileContent[:200])   # extracted text
print(dto.prompt.prompt)       # "What are the risks?"
```

**Dependencies:** `PyPDF2`, `src.config.dtos` (`UserPromptDTO`, `DataUploadDTO`)

---

### `rag.py`

**Class: `PromptSpecificRetriever`**
Splits text into overlapping chunks and retrieves the most relevant ones for a query using a hybrid of TF-IDF cosine similarity and n-gram phrase overlap.

**Constructor:**
```python
retriever = PromptSpecificRetriever(
    text="long document text...",
    max_words=60,    # max words per chunk (RAG_MAX_WORDS)
    overlap=15,      # overlap words between chunks (RAG_OVERLAP)
    min_ngram=1,     # minimum phrase n-gram size
    max_ngram=3      # maximum phrase n-gram size
)
```

**Key Method:**
```python
chunks: List[RetrievedChunkDTO] = retriever.retrieve(
    query="What is the emergency protocol?",
    top_k=3,              # RAG_DEFAULT_TOP_K
    keyword_weight=0.5    # RAG_KEYWORD_WEIGHT: weight for phrase overlap vs cosine
)
# Each chunk has: chunk_id, score, cosine_sim, phrase_overlap, text
```

**Scoring formula:**
```
combined_score = (1 - keyword_weight) * cosine_sim + keyword_weight * phrase_overlap
```

**Internal methods:**
- `_chunk_text(text, max_words, overlap)` → `List[str]` — sliding-window sentence splitter
- `_extract_phrases(text)` → `Set[str]` — generates all 1–3-gram phrases from text

**Dependencies:** `sklearn` (`TfidfVectorizer`, `cosine_similarity`), `src.config.dtos` (`RetrievedChunkDTO`, `RetrievalResponseDTO`)

---

### `__init__.py`
Exports: `PdfProcessor`, `PromptSpecificRetriever`, `RetrievedChunkDTO`, `RetrievalResponseDTO`

---

## Used By
Nothing in `training/` uses this directly — it's available for downstream applications that want to augment scenario text with retrieved context before passing to the encoder.
