import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transformerAndRL.tokenizer import BPETokenizer, EmbeddingTable
from src.transformerAndRL.transformer import BidirectionalEncoderStack, FinalMLP, mean_pooling
from src.output.optionMatrixConvertor import OptionMatrixConvertor
from src.output.finalHeadAndOutput import FinalHeadAndOutput
from training.DPO import DPOTrainer, DPOSample
from training.RLCD import RLCDTrainer, RLCDSample

# =====================================================================
# ONE-CLICK CONFIGURATION (Modify these variables or just click 'Run')
# =====================================================================
TOKENIZER_RAW_DATA = os.path.join(PROJECT_ROOT, "data", "data.txt")
MODEL_TRAIN_DATA = os.path.join(PROJECT_ROOT, "data", "train_dataset.jsonl")
TOKENIZER_OUT_PATH = os.path.join(PROJECT_ROOT, "model_artifacts", "tokenizer", "tokenizer.json")
CHECKPOINT_OUT_PATH = os.path.join(PROJECT_ROOT, "model_artifacts", "checkpoints", "sankalpa_weights.npz")

TARGET_VOCAB_SIZE: int = 500       # Target vocabulary size for BPE
DPO_EPOCHS: int = 10               # Epochs for Stage 1 DPO preference learning
RLCD_EPOCHS: int = 3               # Epochs for Stage 2 RLCD confidence calibration
DPO_LR: float = 2e-4               # Learning rate for DPO
RLCD_LR: float = 5e-4              # Learning rate for RLCD
TEMPERATURE: float = 0.07          # Softmax temperature


# =====================================================================
# 1. DATA ENCODING & SAMPLE BUILDERS
# =====================================================================

def encode_text_to_context(
    text: str,
    tokenizer: BPETokenizer,
    embedder: EmbeddingTable,
    encoder: BidirectionalEncoderStack
) -> np.ndarray:
    """Encodes scenario text into a 768-dimensional context vector [1, 768]."""
    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = [tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else 1]

    input_ids = np.array([token_ids], dtype=np.int64)
    attention_mask = np.ones((1, len(token_ids)), dtype=np.float32)

    emb = embedder.forward(input_ids)
    hidden = encoder.forward(emb, attention_mask=attention_mask)
    context_vec = mean_pooling(hidden, attention_mask=attention_mask)
    return context_vec.astype(np.float32)


def load_jsonl_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Reads and validates single-JSON-per-line dataset file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num} in '{filepath}' is not valid JSON: {e}")

            for field in ["scenario", "options", "preferred", "rejected"]:
                if field not in data:
                    raise ValueError(f"Line {line_num} missing required field '{field}'")

            options = data["options"]
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError(f"Line {line_num}: 'options' must be a list with at least 2 items")

            pref = data["preferred"]
            rej = data["rejected"]
            if pref not in options:
                raise ValueError(f"Line {line_num}: 'preferred' ('{pref}') not found in 'options'")
            if rej not in options:
                raise ValueError(f"Line {line_num}: 'rejected' ('{rej}') not found in 'options'")
            if pref == rej:
                raise ValueError(f"Line {line_num}: 'preferred' and 'rejected' cannot be identical")

            records.append(data)

    print(f"[✓] Successfully loaded and validated {len(records)} samples from {filepath}")
    return records


def build_dpo_samples(
    records: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    embedder: EmbeddingTable,
    encoder: BidirectionalEncoderStack,
    final_mlp: FinalMLP,
    option_convertor: OptionMatrixConvertor,
    temperature: float = 0.07
) -> List[DPOSample]:
    """Constructs DPOSample objects with precomputed context vectors and initial reference logits."""
    dpo_samples: List[DPOSample] = []
    for rec in records:
        context_vec = encode_text_to_context(rec["scenario"], tokenizer, embedder, encoder)
        options = rec["options"]
        winner_idx = options.index(rec["preferred"])
        loser_idx = options.index(rec["rejected"])

        z = final_mlp.forward(context_vec)
        eps = 1e-12
        z_norm = z / (np.linalg.norm(z, ord=2, axis=-1, keepdims=True) + eps)
        M = option_convertor.convert(options)
        ref_logits = (np.matmul(z_norm, M.T) / temperature).astype(np.float32)

        dpo_samples.append(
            DPOSample(
                context_vector=context_vec,
                action_strings=options,
                winner_idx=winner_idx,
                loser_idx=loser_idx,
                ref_logits=ref_logits
            )
        )
    return dpo_samples


def build_rlcd_samples(
    records: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    embedder: EmbeddingTable,
    encoder: BidirectionalEncoderStack,
    final_mlp: FinalMLP,
    option_convertor: OptionMatrixConvertor
) -> List[RLCDSample]:
    """Constructs RLCDSample objects with target affordance and raw scenario vectors."""
    rlcd_samples: List[RLCDSample] = []
    for rec in records:
        options = rec["options"]
        M = option_convertor.convert(options)

        raw_context = encode_text_to_context(rec["scenario"], tokenizer, embedder, encoder)
        z_neg = final_mlp.forward(raw_context)

        w_idx = options.index(rec["preferred"])
        z_pos = M[w_idx:w_idx + 1]

        rlcd_samples.append(
            RLCDSample(
                z_positive=z_pos.astype(np.float32),
                z_negative=z_neg.astype(np.float32),
                options_matrix=M.astype(np.float32)
            )
        )
    return rlcd_samples


# =====================================================================
# 2. TRAINING PIPELINE CLASS
# =====================================================================

class TrainingPipeline:
    def __init__(
        self,
        final_mlp: FinalMLP,
        option_convertor: OptionMatrixConvertor,
        dpo_lr: float = DPO_LR,
        rlcd_lr: float = RLCD_LR,
        temperature: float = TEMPERATURE
    ):
        self.final_mlp = final_mlp
        self.option_convertor = option_convertor
        self.temperature = temperature
        self.dpo_trainer = DPOTrainer(final_mlp, option_convertor, lr=dpo_lr, temperature=temperature)
        self.rlcd_trainer = RLCDTrainer(lr=rlcd_lr, temperature=temperature)

    def run_stage1_dpo(self, dataset: List[DPOSample], epochs: int = DPO_EPOCHS) -> List[float]:
        print(f"\n=======================================================")
        print(f"🚀 Launching Stage 1: DPO Preference Training")
        print(f"   Samples: {len(dataset)} | Epochs: {epochs} | LR: {self.dpo_trainer.optimizer.lr}")
        print(f"=======================================================")
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            total_margin = 0.0
            for step, sample in enumerate(dataset):
                loss, margin = self.dpo_trainer.train_step(sample)
                total_loss += loss
                total_margin += margin

            avg_loss = total_loss / max(len(dataset), 1)
            avg_margin = total_margin / max(len(dataset), 1)
            history.append(avg_loss)
            print(f" [Stage 1 - Epoch {epoch + 1}/{epochs}] Mean Loss: {avg_loss:.4f} | Mean Margin: {avg_margin:+.4f}")
        return history

    def run_stage2_rlcd(self, dataset: List[RLCDSample], epochs: int = RLCD_EPOCHS) -> List[float]:
        print(f"\n=======================================================")
        print(f"🎯 Launching Stage 2: RLCD Domain Calibration")
        print(f"   Samples: {len(dataset)} | Epochs: {epochs} | LR: {self.rlcd_trainer.optimizer.lr}")
        print(f"=======================================================")
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for step, sample in enumerate(dataset):
                loss = self.rlcd_trainer.train_step(sample)
                total_loss += loss

            avg_loss = total_loss / max(len(dataset), 1)
            history.append(avg_loss)
            print(f" [Stage 2 - Epoch {epoch + 1}/{epochs}] Calibration Loss: {avg_loss:.4f}")
        return history

    def save_checkpoint(
        self,
        filepath: str,
        encoder: Optional[BidirectionalEncoderStack] = None,
        embedder: Optional[EmbeddingTable] = None
    ) -> None:
        """Saves model weights and calibration heads to a compressed npz archive."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        save_dict = {
            "mlp_W_gate": self.final_mlp.W_gate,
            "mlp_b_gate": self.final_mlp.b_gate,
            "mlp_W_up": self.final_mlp.W_up,
            "mlp_b_up": self.final_mlp.b_up,
            "mlp_W_down": self.final_mlp.W_down,
            "mlp_b_down": self.final_mlp.b_down,
            "opt_W_proj": self.option_convertor.W_proj,
            "opt_b_proj": self.option_convertor.b_proj,
            "cal_W": self.rlcd_trainer.W_cal,
            "cal_b": self.rlcd_trainer.b_cal
        }
        if encoder is not None:
            save_dict.update(encoder.get_weights())
        if embedder is not None:
            save_dict.update(embedder.get_weights())

        np.savez_compressed(filepath, **save_dict)
        print(f"[✓] Checkpoint successfully saved to: {filepath}")

    def load_checkpoint(
        self,
        filepath: str,
        encoder: Optional[BidirectionalEncoderStack] = None,
        embedder: Optional[EmbeddingTable] = None
    ) -> None:
        """Loads weights from disk."""
        data = np.load(filepath)
        self.final_mlp.W_gate = data["mlp_W_gate"]
        self.final_mlp.b_gate = data["mlp_b_gate"]
        self.final_mlp.W_up = data["mlp_W_up"]
        self.final_mlp.b_up = data["mlp_b_up"]
        self.final_mlp.W_down = data["mlp_W_down"]
        self.final_mlp.b_down = data["mlp_b_down"]
        self.option_convertor.W_proj = data["opt_W_proj"]
        self.option_convertor.b_proj = data["opt_b_proj"]
        self.rlcd_trainer.W_cal = data["cal_W"]
        self.rlcd_trainer.b_cal = data["cal_b"]

        if encoder is not None:
            encoder.set_weights(data)
        if embedder is not None and "embedding_weights" in data:
            embedder.set_weights({"embedding_weights": data["embedding_weights"]})

        print(f"[✓] Checkpoint successfully loaded from: {filepath}")

    def predict(
        self,
        scenario_text: str,
        options: List[str],
        tokenizer: BPETokenizer,
        embedder: EmbeddingTable,
        encoder: BidirectionalEncoderStack
    ):
        """Runs calibrated inference for a scenario text and candidate options list."""
        context_vec = encode_text_to_context(scenario_text, tokenizer, embedder, encoder)
        z = self.final_mlp.forward(context_vec)

        # Apply Stage 2 calibration
        z_cal = np.matmul(z, self.rlcd_trainer.W_cal) + self.rlcd_trainer.b_cal
        options_matrix = self.option_convertor.convert(options)

        final_head = FinalHeadAndOutput(temperature=self.temperature)
        decision_result = final_head.forward(
            situation_vector=z_cal,
            options_matrix=options_matrix,
            candidate_strings=options
        )
        return decision_result


# =====================================================================
# 3. ONE-CLICK MAIN PIPELINE
# =====================================================================

def main():
    print("\n" + "=" * 65)
    print("      SANKALPA (110M): END-TO-END ONE-CLICK TRAINING")
    print("=" * 65)

    # Set fixed seed for deterministic reproducibility across all runs
    SEED = 42
    np.random.seed(SEED)

    if not os.path.exists(MODEL_TRAIN_DATA):
        raise FileNotFoundError(
            f"Model training dataset not found at: '{MODEL_TRAIN_DATA}'. "
            f"Please create '{MODEL_TRAIN_DATA}' in JSONL format."
        )

    # 1. Tokenizer Setup (Load if exists, otherwise train)
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_OUT_PATH):
        print(f"\n[Step 1/5] Loading existing BPETokenizer from: {TOKENIZER_OUT_PATH}...")
        tokenizer.load_from_json(TOKENIZER_OUT_PATH)
        print(f"[✓] Tokenizer loaded (Vocab Size: {tokenizer.vocab_size})")
    else:
        if not os.path.exists(TOKENIZER_RAW_DATA):
            raise FileNotFoundError(
                f"Tokenizer raw data file not found at: '{TOKENIZER_RAW_DATA}'. "
                f"Please create '{TOKENIZER_RAW_DATA}' with your raw corpus text."
            )
        print(f"\n[Step 1/5] Training BPETokenizer on raw text: {TOKENIZER_RAW_DATA}...")
        tokenizer.train_from_file(TOKENIZER_RAW_DATA, target_vocab_size=TARGET_VOCAB_SIZE)
        tokenizer.save(TOKENIZER_OUT_PATH)
        print(f"[✓] Tokenizer trained (Vocab Size: {tokenizer.vocab_size}) and saved to: {TOKENIZER_OUT_PATH}")

    # 2. Synchronize Embedder and Model Components
    print(f"\n[Step 2/5] Initializing synchronized model architecture (seed={SEED})...")
    embedder = EmbeddingTable.from_tokenizer(tokenizer, seed=SEED)
    encoder = BidirectionalEncoderStack(seed=SEED)
    final_mlp = FinalMLP(seed=SEED)
    option_convertor = OptionMatrixConvertor(tokenizer=tokenizer, embedder=embedder, seed=SEED)

    pipeline = TrainingPipeline(
        final_mlp=final_mlp,
        option_convertor=option_convertor,
        dpo_lr=DPO_LR,
        rlcd_lr=RLCD_LR,
        temperature=TEMPERATURE
    )

    # 3. Load & Build Datasets from JSONL
    print(f"\n[Step 3/5] Loading JSONL training records from: {MODEL_TRAIN_DATA}...")
    records = load_jsonl_dataset(MODEL_TRAIN_DATA)

    print("   Building Stage 1 DPO samples (Context Embeddings & Action Matrices)...")
    dpo_samples = build_dpo_samples(
        records, tokenizer, embedder, encoder, final_mlp, option_convertor, temperature=TEMPERATURE
    )

    # 4. Run Stage 1 (DPO) and Stage 2 (RLCD)
    print(f"\n[Step 4/5] Executing Model Training...")
    pipeline.run_stage1_dpo(dpo_samples, epochs=DPO_EPOCHS)

    print("\n   Constructing calibrated Stage 2 RLCD samples using trained DPO representations...")
    rlcd_samples = build_rlcd_samples(
        records, tokenizer, embedder, encoder, final_mlp, option_convertor
    )
    pipeline.run_stage2_rlcd(rlcd_samples, epochs=RLCD_EPOCHS)

    # 5. Save Checkpoint (Including Backbone & Embedder)
    print(f"\n[Step 5/5] Preserving model weights checkpoint...")
    pipeline.save_checkpoint(CHECKPOINT_OUT_PATH, encoder=encoder, embedder=embedder)

    # 6. Post-Training Validation Decision Test
    print(f"\n=======================================================")
    print("🎉 Post-Training Sanity Inference Test")
    print("=======================================================")
    first_record = records[0]
    test_scenario = first_record["scenario"]
    test_options = first_record["options"]

    result = pipeline.predict(test_scenario, test_options, tokenizer, embedder, encoder)

    print(f"Scenario: {test_scenario[:80]}...")
    print(f"🏆 Winning Decision : {result.selected_action.text}")
    print(f"📊 Confidence Score : {result.selected_action.confidence:.2%}")
    print(f"📈 Confidence Margin: {result.selected_action.margin_over_runner_up:.4f}")
    print("\nRankings:")
    for rank in result.rankings:
        print(f"  {rank.rank}. [{rank.probability:.2%}] {rank.text}")

    print("\n[✓] All training stages completed successfully!")


if __name__ == "__main__":
    main()