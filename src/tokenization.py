import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

class TokenizerNandi:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        # 1. Instantiate a clean Byte-Level BPE Core
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        # Explicitly define special tokens as object properties for your dataset loader
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.thought_token = "<|thought|>" # Critical for your reasoning traces
        # Cached IDs mapped cleanly for your PyTorch layers
        self.pad_token_id = None
        self.eos_token_id = None
        self.thought_token_id = None

    def train(self, dataset_path: str, save_dir: str):
        """Analyzes raw business text to build your 50,257 vocabulary map from scratch."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Raw dataset file missing at: '{dataset_path}'")
        os.makedirs(save_dir, exist_ok=True)

        # FIX 1: Use Byte-Level Pre-Tokenizer instead of Whitespace to handle complex strings natively
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Setup the specialized trainer configuration matching your model limits
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=[self.unk_token, self.pad_token, self.eos_token, self.thought_token]
        )
        print(f"--- Training TokenizerNandi [Vocab Size: {self.vocab_size}] ---")
        self.tokenizer.train(files=[dataset_path], trainer=trainer)
        # FIX 2: Attach the corresponding Byte-Level Decoder to restore normal spaces perfectly
        self.tokenizer.decoder = decoders.ByteLevel()
        # Save the single-file layout configuration
        save_path = os.path.join(save_dir, "tokenizer.json")
        self.tokenizer.save(save_path)
        print(f"Tokenizer configuration saved to: {save_path}")
        # Anchor your token IDs to properties
        self._bind_special_tokens()

    def load(self, save_dir: str):
        """Loads your existing custom vocabulary maps back into this runtime object."""
        load_path = os.path.join(save_dir, "tokenizer.json")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Saved tokenizer file not found at: '{load_path}'")
        self.tokenizer = Tokenizer.from_file(load_path)
        self._bind_special_tokens()

    def _bind_special_tokens(self):
        """Internal routine to link special text tags directly with matrix indices."""
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.thought_token_id = self.tokenizer.token_to_id(self.thought_token)

    def encode(self, text: str) -> list:
        """Transforms raw business text into matrix-ready integer IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        """Translates integer sequences from your model outputs back into readable text."""
        return self.tokenizer.decode(ids)


# System-level verification routing check
if __name__ == "__main__":
    import shutil
    # Setup clean sandbox directories
    os.makedirs("data", exist_ok=True)
    MOCK_DATA = "data/business_dataset.txt" #Will enter the proper one in the future.
    SAVE_DIR = "model_artifacts/tokenizer"#Will enter the proper one in the future.
    with open(MOCK_DATA, "w", encoding="utf-8") as f:
        f.write("DataSyncProtocol optimized transaction pathways inside our EnterpriseReasoningEngine.")

    # 1. Initialize and train the object
    t_nandi = TokenizerNandi(vocab_size=50257)
    t_nandi.train(MOCK_DATA, SAVE_DIR)
    # 2. Test operational parameters
    sample = "Execute DataSyncProtocol."
    token_ids = t_nandi.encode(sample)
    decoded_text = t_nandi.decode(token_ids)
    print("\n" + "="*40 + "\nPROTOTYPE VALIDATION DIAGNOSTICS:\n" + "="*40)
    print(f"Vector Index Stream : {token_ids}")
    print(f"Restored Text Output: '{decoded_text}'") # Will decode beautifully with spaces
    print(f"Cached Pad Token ID : {t_nandi.pad_token_id}")
    print(f"Cached Thought Tag ID: {t_nandi.thought_token_id}")
    print("="*40)
    # Clean up mock workspace
    if os.path.exists("data"): shutil.rmtree("data")
