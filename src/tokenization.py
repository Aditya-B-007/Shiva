import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

class Config:
    CORPUS_PATH = "/Users/aditya/Desktop/Projects/Shiva/data/data.txt"
    MODEL_ARTIFACTS_PATH = "/Users/aditya/Desktop/Projects/Shiva/model_artifacts/tokeniser/tokeniser.json"

class TokenizerNandi:
    def __init__(self, model_path=Config.MODEL_ARTIFACTS_PATH):
        self.model_path = model_path
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        special_tokens = ["<unk>", "<pad>", "</s>", "<|thought|>"]
    def train(self, corpus_path=Config.CORPUS_PATH):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        trainer = trainers.BpeTrainer(
            special_tokens=["<unk>", "<pad>", "</s>", "<|thought|>"],
            vocab_size=8192,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        def page_iterator():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

        self.tokenizer.train_from_iterator(
            page_iterator(),
            trainer=trainer
        )
        self.tokenizer.save(self.model_path)
    def load(self):
        self.tokenizer = Tokenizer.from_file(self.model_path)
    def encode(self, text):
        return self.tokenizer.encode(text)
    def decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

if __name__ == "__main__":
    corpus_path = Config.CORPUS_PATH
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    tokenizer = TokenizerNandi()
    model_path = Config.MODEL_ARTIFACTS_PATH
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print("Training new tokenizer...")
        tokenizer.train(corpus_path)
    else:
        print("Loading existing tokenizer...")
        tokenizer.load()
    test_texts = [
       "Mothership",
    "Middleman Protocol",
    "EnvironmentMatrix",
    "AdapterTransformation",
    "Fast Decision Node",
    "Guardrail Failure Engine",
    "Actuator Dispatch",
    "reinforcement learning",
    "control policy",
    "state representation",
    "reward function",
    "model predictive control",
    "CPO",
    "TD3",
    "SAC",
    "IQN",
    "RND",
    ]
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Token IDs: {encoded.ids}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded Text: {decoded}")
        assert text == decoded, "Encode-decode mismatch"
    print("\nTokenizer validation passed successfully!")
    print("Vocabulary size:", tokenizer.tokenizer.get_vocab_size())
