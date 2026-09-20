import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

try:
    from src.interfaces import ITrainableTokenizer
    from src.config import VOCAB_SIZE, MIN_FREQUENCY
except (ImportError, ModuleNotFoundError):
    from interfaces import ITrainableTokenizer
    from config import VOCAB_SIZE, MIN_FREQUENCY

DEFAULT_TOKENIZER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "tokeniser", "tokeniser.json")
)


class TokenizerConfig:
    """Filesystem paths used by TokenizerNandi. Override via environment variables."""
    CORPUS_PATH = os.getenv(
        "CORPUS_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.txt"))
    )
    MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", DEFAULT_TOKENIZER_PATH)


class TokenizerNandi(ITrainableTokenizer):
    def __init__(self, modelPath=None):
        if modelPath is None:
            modelPath = TokenizerConfig.MODEL_ARTIFACTS_PATH or DEFAULT_TOKENIZER_PATH
        self.model_path = modelPath
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, corpusPath=TokenizerConfig.CORPUS_PATH) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        trainer = trainers.BpeTrainer(
            special_tokens=["<unk>", "<pad>", "</s>", "<|thought|>", "<image>"],
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        def pageIterator():
            with open(corpusPath, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

        self.tokenizer.train_from_iterator(pageIterator(), trainer=trainer)
        self.tokenizer.save(self.model_path)

    def load(self) -> None:
        self.tokenizer = Tokenizer.from_file(self.model_path)

    def addSpecialTokens(self, tokens) -> int:
        return self.tokenizer.add_special_tokens(tokens)

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, tokenIds, skipSpecialTokens: bool = False) -> str:
        return self.tokenizer.decode(tokenIds, skip_special_tokens=skipSpecialTokens)

    def getVocabSize(self) -> int:
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    corpusPath = TokenizerConfig.CORPUS_PATH
    os.makedirs(os.path.dirname(corpusPath), exist_ok=True)

    tokenizer = TokenizerNandi()
    modelPath = TokenizerConfig.MODEL_ARTIFACTS_PATH
    import sys
    forceRetrain = "--train" in sys.argv or "--force" in sys.argv

    if forceRetrain or not os.path.exists(modelPath) or os.path.getsize(modelPath) == 0:
        tokenizer.train(corpusPath)
    else:
        tokenizer.load()

    testTexts = ["Boy", "Girl", "Happy", "Environment", "Door", "Banana", "Car"]
    for text in testTexts:
        encoded = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Token IDs: {encoded.ids}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded Text: {decoded}")
        assert text == decoded, "Encode-decode mismatch"
    print("Vocabulary size:", tokenizer.getVocabSize())
