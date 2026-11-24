"""
Training loop
Next-token prediction with Adam, gradient clipping, early stopping
and a checkpoint every 10 epochs.
"""

import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from tokenizer import BPETokenizer
from model import SolTransformer, count_parameters


class MemoryDataset(Dataset):
    """
    Dataset from simple text file
    Returns sequences for next-token prediction
    """
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenize ALL texts into one continuous stream
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        # Create sequences
        self.sequences = []
        step_size = seq_len 
        for i in range(0, len(all_tokens) - seq_len - 1, step_size):
            input_seq = all_tokens[i : i + seq_len]
            target_seq = all_tokens[i + 1 : i + seq_len + 1]
            self.sequences.append((input_seq, target_seq))
        
        print(f"[Created {len(self.sequences)} sequences from {len(all_tokens)} tokens]")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch
    
    Args:
        model: the transformer
        dataloader: Training data
        optimizer: Adam optimizer
        criterion: Loss function
        device: cuda/cpu
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(inputs)
        
        # Reshape for loss calculation
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Calculate loss
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (architectural constant for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape for loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(logits, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    """Plot and save training/validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Loss curve saved to {save_path}]")


def generate_sample(model, tokenizer, input_tokens, max_length, device, temperature=1.0):
    """
    Generate text from tokens
    
    Args:
        model: Trained transformer
        tokenizer: BPE tokenizer
        input_tokens: Starting token IDs
        max_length: Max tokens to generate
        device: cuda/cpu
        temperature: sampling temperature
        
    Returns:
        Generated text
    """
    model.eval()
    
    input_ids = torch.tensor([input_tokens]).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature
        )
    
    # Decode
    generated_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def train_sol(
    vocab_size=712,
    d_model=512,
    num_heads=8,
    num_layers=12,
    d_ff=2048,
    seq_len=128,
    batch_size=24,
    epochs=150,
    base_learning_rate=0.0003,
    save_dir='checkpoints'
):
    """
    Main training function
    Reads syn_train.txt / syn_val.txt and tokenizer_syn.json from this folder.

    Args:
        vocab_size: Overwritten by the loaded tokenizer's vocabulary size
        d_model: Model dimension
        num_heads: Attention heads
        num_layers: Transformer layers
        d_ff: Feed-forward dimension
        seq_len: Sequence length for training
        batch_size: Batch size
        epochs: Number of training epochs
        base_learning_rate: Base learning rate
        save_dir: Directory to save checkpoints
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Using device: {device}]")
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load tokenizer
    print("\n[Loading tokenizer...]")
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer_syn.json')
    vocab_size = len(tokenizer.vocab)
    
    # Load texts
    print("[Loading texts...]")
    with open('syn_train.txt', 'r', encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]

    with open('syn_val.txt', 'r', encoding='utf-8') as f:
        val_texts = [line.strip() for line in f if line.strip()]
    
    print(f"[Train texts: {len(train_texts)}]")
    print(f"[Val texts: {len(val_texts)}]")
    
    # Create datasets
    train_dataset = MemoryDataset(train_texts, tokenizer, seq_len)
    val_dataset = MemoryDataset(val_texts, tokenizer, seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n[Creating model...]")
    model = SolTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=512,
        dropout=0.15
    ).to(device)
    
    print(f"[Model parameters: {count_parameters(model):,}]")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_learning_rate, weight_decay=0.001)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {base_learning_rate}")
    print(f"{'='*70}\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = f"{save_dir}/sol_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")
        
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{save_dir}/sol_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved")
        
        print()
    
    # Plot losses
    plot_losses(train_losses, val_losses, f"{save_dir}/loss_curve.png")
    
    # Generate sample
    print("\n" + "="*70)
    print("SAMPLE GENERATION")
    print("="*70)
    
    # Generate from training data (emergent, not hardcoded)
    sample_tokens = train_dataset.sequences[0][0][:10]
    sample_prompt = tokenizer.decode(sample_tokens)
    
    print(f"Prompt (from training): '{sample_prompt}'")
    print("Generated:")
    
    generated = generate_sample(
        model, tokenizer, sample_tokens, 
        max_length=30, device=device
    )
    print(f"  {generated}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {save_dir}/")
    print("="*70)


if __name__ == '__main__':
    train_sol()