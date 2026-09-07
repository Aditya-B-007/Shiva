import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def chat():
    device = get_device()
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    
    # Priority: 1. nandi_r1_reasoning.pt (RL CoT) -> 2. nandi_chat_final.pt -> 3. nandi_final.pt
    r1_ckpt = os.path.join(checkpoint_dir, "nandi_r1_reasoning.pt")
    chat_ckpt = os.path.join(checkpoint_dir, "nandi_chat_final.pt")
    base_ckpt = os.path.join(checkpoint_dir, "nandi_final.pt")

    if os.path.exists(r1_ckpt):
        checkpoint_path = r1_ckpt
        print(">> Loading RL-Trained Reasoning Model (nandi_r1_reasoning.pt)...")
    elif os.path.exists(chat_ckpt):
        checkpoint_path = chat_ckpt
        print(">> Loading SFT Chat Model (nandi_chat_final.pt)...")
    elif os.path.exists(base_ckpt):
        checkpoint_path = base_ckpt
        print(">> Loading Base Pre-trained Model (nandi_final.pt)...")
    else:
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("nandi_") and f.endswith(".pt")]) if os.path.exists(checkpoint_dir) else []
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        else:
            print(f"No trained checkpoint found at: {checkpoint_dir}")
            print("Please run `python3 training/train.py` or `python3 training/finetune_qa.py` first!")
            sys.exit(1)

    print("Loading Tokenizer...")
    tokenizer = TokenizerNandi()
    tokenizer.load()
    vocab_size = tokenizer.get_vocab_size()

    print(f"Loading Model from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = TransformerModel(
        ntoken=vocab_size,
        ninp=512,
        nhead=8,
        nhid=2048,
        nlayers=8,
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Nandi SLM is ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("\nUser > ")
            if prompt.strip().lower() in ["exit", "quit"]:
                break
            if not prompt.strip():
                continue

            # Format with instruction-tuning template
            formatted_prompt = f"User: {prompt}\nAssistant: <|thought|>\n"
            encoded = tokenizer.encode(formatted_prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

            print("\nNandi > ", end="", flush=True)
            eos_id = tokenizer.tokenizer.token_to_id("</s>")
            output_ids = model.generate(
                input_ids,
                max_new_tokens=450,
                temperature=0.35,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.15,
                eos_token_id=eos_id
            )

            new_tokens = output_ids[0, input_ids.size(1):].tolist()
            if eos_id is not None and eos_id in new_tokens:
                new_tokens = new_tokens[:new_tokens.index(eos_id)]

            raw_response = tokenizer.decode(new_tokens)
            
            # Format Chain of Thought cleanly in the terminal
            if "<|thought|>" in raw_response:
                parts = raw_response.split("<|thought|>")
                thought_content = parts[0].strip()
                final_answer = parts[1].strip() if len(parts) > 1 else ""
                print(f"[🧠 Thinking Process]\n{thought_content}\n\n[💡 Final Specification]\n{final_answer}")
            else:
                print(raw_response)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break

if __name__ == "__main__":
    chat()
