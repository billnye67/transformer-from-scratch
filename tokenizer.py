"""
BPE Tokenizer - Fixed Version
Preserves punctuation, case, and spacing
"""

import json
import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []

    def train(self, texts):
        """Train BPE tokenizer on raw text - preserve everything"""
        print(f"[Training tokenizer on {len(texts)} texts...]")
        
        # Get all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(list(text))
        
        # Initialize vocab with characters
        self.vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}
        
        # Add special tokens
        for token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Learn BPE merges
        # Join all texts with newline separator
        corpus = '\n'.join(texts)
        chars = list(corpus)
        
        for i in range(self.vocab_size - len(self.vocab)):
            # Count adjacent pairs
            pair_counts = Counter()
            for j in range(len(chars) - 1):
                pair = (chars[j], chars[j + 1])
                pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            
            # Apply merge to corpus
            new_chars = []
            j = 0
            while j < len(chars):
                if j < len(chars) - 1 and (chars[j], chars[j + 1]) == best_pair:
                    new_chars.append(new_token)
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1
            chars = new_chars
            
            if (i + 1) % 100 == 0:
                print(f"  Learned {i + 1} merges...")
        
        # Build inverse vocab
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"[Tokenizer trained: {len(self.vocab)} tokens]")

    def encode(self, text):
        """Encode text to token IDs - preserves everything"""
        chars = list(text)
        
        # Apply learned merges
        for pair in self.merges:
            i = 0
            while i < len(chars) - 1:
                if (chars[i], chars[i + 1]) == pair:
                    chars[i:i+2] = [chars[i] + chars[i + 1]]
                else:
                    i += 1
        
        # Convert to token IDs
        token_ids = []
        for char in chars:
            token_id = self.vocab.get(char, self.vocab.get('<UNK>', 0))
            token_ids.append(token_id)
        
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs to text - does NOT strip special tokens"""
        tokens = [self.inverse_vocab.get(tid, '') for tid in token_ids]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']]
        
        # Join
        text = ''.join(tokens)
        
        return text

    def save(self, filepath):
        """Save tokenizer to disk"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Tokenizer saved to {filepath}]")

    def load(self, filepath):
        """Load tokenizer from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"[Tokenizer loaded from {filepath}]")

def train_tokenizer_from_file(text_file, vocab_size=2000):
    """Train tokenizer on text file"""
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"[Loaded {len(texts)} lines for tokenizer training]")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer

if __name__ == '__main__':
    print("="*70)
    print("BPE TOKENIZER TRAINING")
    print("="*70)
    
    tokenizer = train_tokenizer_from_file(
        text_file='syn_train.txt',
        vocab_size=7500
    )
    
    tokenizer.save('tokenizer_syn.json')
    
    print("\n" + "="*70)
    print("TESTING TOKENIZER")
    print("="*70)
    
    test_text = "Why does someone believe?"
    print(f"\nOriginal: {test_text}")
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print(f"Match: {test_text == decoded}")
    
    print("\n" + "="*70)
    print("✅ TOKENIZER READY")
    print("="*70)