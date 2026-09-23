from typing import List, Optional, Any
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
    """Detailed score evaluation for an individual candidate option."""
    rank: int = Field(..., ge=1, description="1-based rank (1 is best)")
    index: int = Field(..., ge=0, description="Original index k in the [1, K] vector")
    id: Optional[str] = Field(default=None, description="Unique identifier for the option")
    text: str = Field(..., description="Action description or option text")
    probability: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence probability P(k)")
    logit: float = Field(..., description="Temperature-scaled logit score")
    cosine_sim: float = Field(..., description="Cosine similarity or alignment score with situation vector")


class SelectedActionDTO(BaseModel):
    """Summary of the winning / selected decision from the [1, K] distribution."""
    rank: int = Field(default=1, description="Rank 1 for winning choice")
    index: int = Field(..., ge=0, description="Winning index in the [1, K] options matrix")
    id: Optional[str] = Field(default=None, description="Option identifier")
    text: Optional[str] = Field(default=None, description="Action description text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Winning option probability")
    logit: float = Field(..., description="Winning option logit")
    margin_over_runner_up: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability difference over 2nd place")


class DecisionMetricsDTO(BaseModel):
    """Information-theoretic and statistical diagnostics of the [1, K] distribution."""
    total_options: int = Field(..., ge=1, description="Total number of evaluated candidates (K)")
    temperature: float = Field(..., gt=0.0, description="Softmax temperature parameter (tau)")
    top_probability: float = Field(..., ge=0.0, le=1.0, description="Highest probability score in [1, K]")
    runner_up_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Second highest probability in [1, K]")
    confidence_margin: Optional[float] = Field(default=None, description="top_probability - runner_up_probability")
    entropy: float = Field(..., ge=0.0, description="Shannon entropy measuring uncertainty across K options")


class DecisionDistributionDTO(BaseModel):
    """Full raw distributions across all K candidate options, preserving [1, K] vector shapes."""
    shape: List[int] = Field(default_factory=lambda: [1, 0], description="Shape of the tensor, e.g. [1, K]")
    probabilities_matrix: List[List[float]] = Field(..., description="Exact 2D matrix of shape [1, K]")
    logits_matrix: List[List[float]] = Field(..., description="Exact 2D matrix of shape [1, K]")
    probabilities: List[float] = Field(..., description="1D vector of length K (flattened from [1, K])")
    logits: List[float] = Field(..., description="1D vector of length K (flattened from [1, K])")
    cosine_similarities: List[float] = Field(..., description="1D vector of length K")


class FinalDecisionOutputDTO(BaseModel):
    """Hierarchical root DTO aggregating decision result, rankings, metrics, and distributions."""
    query_id: Optional[str] = Field(default=None, description="Correlates the decision to user query/prompt")
    selected_action: SelectedActionDTO = Field(..., description="Winning candidate decision")
    rankings: List[OptionScoreDTO] = Field(..., description="All candidate options sorted by rank")
    metrics: DecisionMetricsDTO = Field(..., description="Decision confidence and distribution metrics")
    distribution: DecisionDistributionDTO = Field(..., description="Full [1, K] probability and logit distributions")


# Backwards compatibility alias
DecisionResultDTO = FinalDecisionOutputDTO


# =====================================================================
# 5. Reinforcement Learning Policy DTOs
# =====================================================================

class ScoredActionDTO(BaseModel):
    """Scored candidate action produced during RL policy evaluation."""
    rank: int = Field(..., ge=1, description="1-based rank")
    action_index: int = Field(..., ge=0, description="Index in options matrix")
    action_text: str = Field(..., description="Action text description")
    probability: float = Field(..., ge=0.0, le=1.0, description="Action probability")
    logit: float = Field(..., description="Action logit score")


class DecisionOutputDTO(BaseModel):
    """Output from RLPolicy evaluation containing decision probabilities and ranked actions."""
    model_config = {"arbitrary_types_allowed": True}

    probabilities: Any = Field(..., description="Action probability distribution (array or list)")
    logits: Any = Field(..., description="Action logits (array or list)")
    best_action_idx: int = Field(..., ge=0, description="Winning action index")
    best_action_text: Optional[str] = Field(default=None, description="Winning action text")
    ranked_actions: List[ScoredActionDTO] = Field(default_factory=list, description="Ranked list of evaluated actions")