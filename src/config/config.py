from dataclasses import dataclass, field, asdict
from typing import Dict, Any


# =====================================================================
# 1. TOP-LEVEL ARCHITECTURAL CONSTANTS
# =====================================================================

# Model Dimensions
HIDDEN_DIM: int = 768                     # Feature width d_model
NUM_LAYERS: int = 12                     # 12-layer bidirectional encoder stack
NUM_HEADS: int = 12                      # Multi-head attention heads
HEAD_DIM: int = 64                       # Per-head width (768 // 12 = 64)
FFN_DIM: int = 3072                      # SwiGLU expansion width (4 * 768)
MAX_CONTEXT_LENGTH: int = 512            # Maximum context window length
SITUATION_DIM: int = 512                 # Compressed decision state width z in R^512

# Mathematical & Normalization Primitives
ROPE_THETA: float = 10000.0              # RoPE base frequency theta
LAYER_NORM_EPS: float = 1e-12            # Epsilon for LayerNormalization
SILU_CLIP_BOUND: float = 30.0            # Bounds for SiLU activation numerical stability

# Tokenizer & Vocabulary
PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
BOS_TOKEN: str = "<s>"
EOS_TOKEN: str = "</s>"
PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3
BYTE_VOCAB_SIZE: int = 256
BASE_VOCAB_SIZE: int = 260               # 4 special tokens + 256 raw bytes
DEFAULT_VOCAB_SIZE: int = 30522          # 110M architecture spec vocabulary size
MIN_MERGE_FREQUENCY: int = 2             # Minimum BPE pair frequency for merging

# RAG & Chunking
RAG_MAX_WORDS: int = 60                  # Chunk word budget
RAG_OVERLAP: int = 15                    # Overlap words between consecutive segments
RAG_MIN_NGRAM: int = 1                   # Minimum n-gram size for phrase matching
RAG_MAX_NGRAM: int = 3                   # Maximum n-gram size for phrase matching
RAG_DEFAULT_TOP_K: int = 3               # Default ranked retrieved chunks
RAG_KEYWORD_WEIGHT: float = 0.5          # Balance between TF-IDF cosine and phrase overlap
RAG_SUBLINEAR_TF: bool = True            # Sublinear term frequency scaling 1 + log(tf)

# Option Matrix Converter
OPTION_EMBED_DIM: int = 768              # Shared embedding input width
OPTION_OUT_DIM: int = 512                # Projected dimension matching situation vector
OPTION_PROJ_SEED: int = 42               # Seed for projection matrix initialization
OPTION_NORM_EPS: float = 1e-12           # Epsilon for S^511 sphere normalization

# Final Head & Output
DEFAULT_TEMPERATURE: float = 0.07        # Softmax temperature tau
SIMILARITY_EPS: float = 1e-12            # Normalization epsilon for situation vector
MIN_CANDIDATE_OPTIONS: int = 2           # Minimum candidate actions
MAX_CANDIDATE_OPTIONS: int = 32          # Maximum candidate actions

# RL Policy & MCTS
MCTS_NUM_SIMULATIONS: int = 64           # MCTS rollout simulations per decision
MCTS_C_PUCT: float = 1.414               # Exploration constant sqrt(2)
MCTS_DIRICHLET_ALPHA: float = 0.3        # Dirichlet prior concentration parameter
MCTS_DIRICHLET_WEIGHT: float = 0.25      # Dirichlet exploration noise mixture weight


# =====================================================================
# 2. STRUCTURED CONFIGURATION DATACLASSES
# =====================================================================

@dataclass
class TransformerConfig:
    """Hyperparameters for the 12-layer Transformer encoder and Situation MLP."""
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    head_dim: int = HEAD_DIM
    ffn_dim: int = FFN_DIM
    max_context_length: int = MAX_CONTEXT_LENGTH
    situation_dim: int = SITUATION_DIM
    rope_theta: float = ROPE_THETA
    layer_norm_eps: float = LAYER_NORM_EPS
    silu_clip_bound: float = SILU_CLIP_BOUND

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerConfig:
    """Settings for Byte-Pair Encoding tokenizer and Embedding Table."""
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN
    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN
    pad_token_id: int = PAD_TOKEN_ID
    unk_token_id: int = UNK_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    byte_vocab_size: int = BYTE_VOCAB_SIZE
    base_vocab_size: int = BASE_VOCAB_SIZE
    default_vocab_size: int = DEFAULT_VOCAB_SIZE
    min_merge_frequency: int = MIN_MERGE_FREQUENCY
    hidden_dim: int = HIDDEN_DIM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGConfig:
    """Settings for text segmentation and hybrid TF-IDF phrase retrieval."""
    max_words: int = RAG_MAX_WORDS
    overlap: int = RAG_OVERLAP
    min_ngram: int = RAG_MIN_NGRAM
    max_ngram: int = RAG_MAX_NGRAM
    default_top_k: int = RAG_DEFAULT_TOP_K
    keyword_weight: float = RAG_KEYWORD_WEIGHT
    sublinear_tf: bool = RAG_SUBLINEAR_TF

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptionConverterConfig:
    """Configuration for OptionMatrixConvertor projecting candidate strings onto S^511."""
    hidden_dim: int = OPTION_EMBED_DIM
    out_dim: int = OPTION_OUT_DIM
    seed: int = OPTION_PROJ_SEED
    eps: float = OPTION_NORM_EPS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalHeadConfig:
    """Configuration for FinalHeadAndOutput scoring and distribution head."""
    temperature: float = DEFAULT_TEMPERATURE
    eps: float = SIMILARITY_EPS
    min_options: int = MIN_CANDIDATE_OPTIONS
    max_options: int = MAX_CANDIDATE_OPTIONS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLPolicyConfig:
    """Configuration for Reinforcement Learning policy and MCTS reasoning."""
    temperature: float = DEFAULT_TEMPERATURE
    num_simulations: int = MCTS_NUM_SIMULATIONS
    c_puct: float = MCTS_C_PUCT
    dirichlet_alpha: float = MCTS_DIRICHLET_ALPHA
    dirichlet_weight: float = MCTS_DIRICHLET_WEIGHT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 3. UNIFIED MASTER CONFIGURATION CONTAINER
# =====================================================================

@dataclass
class ModelConfig:
    """Unified master configuration aggregating all model and pipeline subsystem settings."""
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    option_converter: OptionConverterConfig = field(default_factory=OptionConverterConfig)
    final_head: FinalHeadConfig = field(default_factory=FinalHeadConfig)
    rl_policy: RLPolicyConfig = field(default_factory=RLPolicyConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default global instance
default_config = ModelConfig()
