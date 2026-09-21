from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. Data Upload & User Prompt DTOs
# =====================================================================

class UserPromptDTO(BaseModel):
    """Represents an instruction or question provided by the user."""
    prompt: Optional[str] = Field(
        default=None,
        description="User instruction or prompt"
    )


class DataUploadDTO(BaseModel):
    """Represents uploaded file data alongside optional user prompt context."""
    fileContent: Optional[str] = Field(
        default=None,
        description="Content of the file to upload (the extracted text)"
    )
    fileName: Optional[str] = Field(
        default=None,
        description="Name of the file to be uploaded"
    )
    prompt: Optional[UserPromptDTO] = Field(
        default=None,
        description="Nested prompt object"
    )


# =====================================================================
# 2. Candidate Options & Matrix DTOs
# =====================================================================

class CandidateOptionDTO(BaseModel):
    """Represents an individual candidate decision/action option."""
    id: str = Field(
        ...,
        description="Unique key to identify the option (e.g., 'opt_1')"
    )
    text: str = Field(
        ...,
        max_length=200,
        description="The English action description"
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional tag (e.g., 'emergency', 'maintenance')"
    )


class OptionMatrixRequestDTO(BaseModel):
    """Request payload containing candidate options correlated to a query."""
    query_id: str = Field(
        ...,
        description="Correlates these options to the user prompt"
    )
    options: List[CandidateOptionDTO] = Field(
        ...,
        min_length=2,
        max_length=32,
        description="List of candidate actions to score (minimum 2, maximum 32)"
    )


# =====================================================================
# 3. Retrieval-Augmented Generation (RAG) DTOs
# =====================================================================

class RetrievedChunkDTO(BaseModel):
    """A single ranked text chunk retrieved based on hybrid relevance scores."""
    chunk_id: int = Field(
        ...,
        description="Unique index of the retrieved chunk"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Combined hybrid score between 0.0 and 1.0"
    )
    cosine_sim: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between 0.0 and 1.0"
    )
    phrase_overlap: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Length-weighted phrase overlap score between 0.0 and 1.0"
    )
    text: str = Field(
        ...,
        description="Extracted chunk text content"
    )


class RetrievalResponseDTO(BaseModel):
    """Container holding retrieval results for a specific query."""
    query: str = Field(
        ...,
        description="The query string used for retrieval"
    )
    chunks: List[RetrievedChunkDTO] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks"
    )


# =====================================================================
# 4. Final Head & Output Decision DTOs
# =====================================================================

class OptionScoreDTO(BaseModel):
    """Detailed score evaluation for an individual option."""
    id: str = Field(
        ...,
        description="Unique identifier for the option"
    )
    text: str = Field(
        ...,
        description="The English action description or option text"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence probability of the option (between 0.0 and 1.0)"
    )
    raw_score: float = Field(
        ...,
        description="Raw logit or model score before normalization"
    )


class DecisionResultDTO(BaseModel):
    """Final decision outcome produced by the scoring head."""
    query_id: str = Field(
        ...,
        description="Correlates the decision to the originating query/prompt"
    )
    selected_option_id: str = Field(
        ...,
        description="ID of the highest scoring or selected option"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall decision confidence score (between 0.0 and 1.0)"
    )
    rankings: List[OptionScoreDTO] = Field(
        ...,
        description="Ranked list of option scores"
    )
