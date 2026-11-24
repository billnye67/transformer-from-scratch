# debug_overfit.py
import torch
from model import SolTransformer, count_parameters
from tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load('tokenizer_syn.json')

# Tiny dataset
tiny_texts = [
    "Hello, how are you?",
    "I'm doing well, thanks!",
    "What are your plans today?",
    "I plan to read a book.",
    "That sounds relaxing.",
] * 20

# Small model, NO dropout
model = SolTransformer(
    vocab_size=len(tokenizer.vocab),
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    max_seq_len=128,
    dropout=0.0
).to('cuda')

print(f"Parameters: {count_parameters(model):,}")

# Tokenize
all_tokens = []
for text in tiny_texts:
    all_tokens.extend(tokenizer.encode(text))

# Create one batch
seq_len = 64
input_seq = torch.tensor(all_tokens[:seq_len]).unsqueeze(0).to('cuda')
target_seq = torch.tensor(all_tokens[1:seq_len+1]).unsqueeze(0).to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nOverfitting on 1 batch...")

for i in range(500):
    optimizer.zero_grad()
    logits = model(input_seq)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_seq.view(-1)
    )
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}")

print("\nIf loss is NOT near 0.01, architecture is broken.")

# Test generation
model.eval()
prompt = "Hello, how"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens]).to('cuda')

with torch.no_grad():
    output = model.generate(input_ids, max_length=30, temperature=0.5)

print(f"\nPrompt: '{prompt}'")
print(f"Generated: '{tokenizer.decode(output[0].tolist())}'")