import re
from typing import List, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from src.config.dtos import RetrievedChunkDTO, RetrievalResponseDTO
except ImportError:
    from Shiva.src.config.dtos import RetrievedChunkDTO, RetrievalResponseDTO

__all__ = ["RetrievedChunkDTO", "RetrievalResponseDTO", "PromptSpecificRetriever"]

class PromptSpecificRetriever:
    def __init__(
        self,
        text: str,
        max_words: int = 60,
        overlap: int = 15,
        min_ngram: int = 1,
        max_ngram: int = 3, 
    ):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.chunks = self._chunk_text(text, max_words=max_words, overlap=overlap)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        if self.chunks:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(self.min_ngram, self.max_ngram),
                    sublinear_tf=True
                )
                matrix = vectorizer.fit_transform(self.chunks)
                # Check if non-empty vocabulary was learned
                if getattr(vectorizer, "vocabulary_", None):
                    self.vectorizer = vectorizer
                    self.tfidf_matrix = matrix
            except ValueError:
                # Occurs if documents only contain stop words or empty terms
                self.vectorizer = None
                self.tfidf_matrix = None

    @staticmethod
    def _chunk_text(text: str, max_words: int = 60, overlap: int = 15) -> List[str]:
        if not text or not text.strip():
            return []
        raw_segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        normalized_segments: List[str] = []
        for segment in raw_segments:
            words = segment.split()
            if len(words) <= max_words:
                normalized_segments.append(segment)
            else:
                step = max(1, max_words - overlap)
                for i in range(0, len(words), step):
                    sub_words = words[i : i + max_words]
                    if sub_words:
                        normalized_segments.append(" ".join(sub_words))

        chunks: List[str] = []
        current: List[str] = []
        curr_len = 0

        for segment in normalized_segments:
            s_len = len(segment.split())
            if curr_len + s_len > max_words and current:
                joined_current = " ".join(current)
                chunks.append(joined_current)
                overlap_words = joined_current.split()[-overlap:] if overlap > 0 else []
                overlap_text = " ".join(overlap_words)
                current = [overlap_text, segment] if overlap_text else [segment]
                curr_len = len(overlap_words) + s_len
            else:
                current.append(segment)
                curr_len += s_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _extract_phrases(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        phrases: Set[str] = set()
        n = len(tokens)
        for size in range(self.min_ngram, self.max_ngram + 1):
            for i in range(n - size + 1):
                phrases.add(" ".join(tokens[i : i + size]))
        return phrases

    def retrieve(self, query: str, top_k: int = 3, keyword_weight: float = 0.5) -> List[RetrievedChunkDTO]:
        if not self.chunks or not query or not query.strip():
            return []

        cosine_scores = [0.0] * len(self.chunks)
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten().tolist()
            except Exception:
                cosine_scores = [0.0] * len(self.chunks)

        query_phrases = self._extract_phrases(query)
        total_query_weight = sum(len(p.split()) for p in query_phrases) if query_phrases else 0.0

        final_results: List[RetrievedChunkDTO] = []

        for idx, chunk in enumerate(self.chunks):
            cos_score = max(0.0, min(1.0, float(cosine_scores[idx])))

            if total_query_weight == 0.0:
                kw_score = 0.0
            else:
                chunk_phrases = self._extract_phrases(chunk)
                matches = query_phrases.intersection(chunk_phrases)
                matched_weight = sum(len(p.split()) for p in matches)
                kw_score = max(0.0, min(1.0, matched_weight / total_query_weight))

            combined = ((1.0 - keyword_weight) * cos_score) + (keyword_weight * kw_score)
            combined = max(0.0, min(1.0, combined))

            final_results.append(
                RetrievedChunkDTO(
                    chunk_id=idx,
                    score=round(combined, 4),
                    cosine_sim=round(cos_score, 4),
                    phrase_overlap=round(kw_score, 4),
                    text=chunk
                )
            )

        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]