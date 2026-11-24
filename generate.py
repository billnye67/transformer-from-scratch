"""
Load a checkpoint and print sample text.

Usage:
    python generate.py --checkpoint checkpoints/sol_best.pt --tokenizer tokenizer_syn.json
    python generate.py --checkpoint checkpoints/sol_best.pt --tokenizer tokenizer_syn.json \
        --prompt "Do you like hiking in forests?" --tokens 60 --temperature 0.7 --top_k 40

The model shape (width, layers, feed-forward size, vocabulary) is read from the
checkpoint. The number of attention heads is not stored in the weights, so pass
--heads if it is not the default of 8.
"""

import argparse
import torch

from tokenizer import BPETokenizer
from model import SolTransformer, count_parameters


def load_model(checkpoint_path, num_heads, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict']

    vocab_size, d_model = state['token_embedding.weight'].shape
    d_ff = state['transformer_blocks.0.feed_forward.linear1.weight'].shape[0]
    num_layers = len({k.split('.')[1] for k in state if k.startswith('transformer_blocks.')})
    # Checkpoints saved before the final layer norm was added do not have it.
    has_final_norm = 'final_norm.weight' in state

    model = SolTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        final_norm=has_final_norm,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    print(f"[Loaded {checkpoint_path}]")
    print(f"[{count_parameters(model):,} parameters, {num_layers} layers, d_model {d_model}, "
          f"vocab {vocab_size}, final norm: {has_final_norm}]")
    if 'val_loss' in checkpoint:
        print(f"[Saved at epoch {checkpoint['epoch'] + 1}, validation loss {checkpoint['val_loss']:.4f}]")
    return model


def main():
    parser = argparse.ArgumentParser(description="Print sample text from a trained checkpoint")
    parser.add_argument('--checkpoint', default='checkpoints/sol_best.pt')
    parser.add_argument('--tokenizer', default='tokenizer_syn.json')
    parser.add_argument('--prompt', action='append',
                        help="Prompt text. Repeat the flag for several prompts.")
    parser.add_argument('--tokens', type=int, default=40, help="Tokens to generate per prompt")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--heads', type=int, default=8, help="Attention heads the model was trained with")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, args.heads, device)

    prompts = args.prompt or [
        "Do you like hiking in forests?",
        "What is your favorite food?",
        "Hello, how are you?",
    ]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        output = model.generate(
            input_ids,
            max_length=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print()
        print(f"Prompt:    {prompt}")
        print(f"Generated: {tokenizer.decode(output[0].tolist())}")


if __name__ == '__main__':
    main()
