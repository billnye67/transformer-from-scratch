# transformer-from-scratch

A small decoder-only transformer language model written by hand in PyTorch, with its own tokenizer and training loop. I built it to learn how the parts work, not to compete with anything.

No pretrained weights and no model libraries are used. The only dependencies are PyTorch and matplotlib (for the loss plot).

## What's in here

| File | What it does |
|---|---|
| `model.py` | The transformer: token embedding, multi-head self-attention, feed-forward layers, pre-layer-norm residual blocks, causal mask, sampling with temperature and top-k. |
| `rope.py` | Rotary position embeddings, applied to the queries and keys inside attention. |
| `tokenizer.py` | A byte-pair-encoding tokenizer in pure Python. It learns merges from raw text and keeps case, punctuation and spacing. |
| `train.py` | Training loop: next-token prediction, Adam, gradient clipping, early stopping, a checkpoint every 10 epochs, and a loss plot. |
| `debug_overfit.py` | Sanity test: a small model has to memorise one batch. If the loss doesn't fall close to zero, something in the model is broken. |
| `generate.py` | Loads a checkpoint and prints sample text. |

## The three models I trained

All three use this model code. Checkpoints are not in the repo because of their size.

| Parameters | Shape | Training text | Tokenizer vocabulary | Best validation loss |
|---|---|---|---|---|
| 4,137,331 | 4 layers, width 256, 4 heads | WikiText-2 (about 10.8 MB) | 1,907 | 4.03 after 20 epochs |
| 7,051,684 | 4 layers, width 256, 4 heads | Synthetic short dialogues (about 4 MB) | 7,588 | 3.91 at epoch 92 |
| 45,516,108 | 12 layers, width 512, 8 heads | The same dialogues | 7,500 | 4.64 at epoch 101 |

Things to know when reading those numbers:

- **The dataset is small.** About 4 MB of text is far too little for a 45.5M-parameter model. Its loss stops improving after roughly 10 epochs and stays flat. More parameters did not help.
- **The losses are not directly comparable.** The first two models used an earlier tokenizer that lowercases text and drops punctuation, which is an easier prediction task. The 45.5M model used the tokenizer in this repo, which keeps everything.
- **Output quality is limited.** The 7M and 45.5M models write short, grammatical-looking dialogue turns that drift off topic within a sentence or two. The 4.1M model mostly produces fragments.
- **The final layer norm in `model.py` was added after these runs.** None of the three models was trained with it. `generate.py` detects older checkpoints and loads them without it.

## Running it

```
pip install -r requirements.txt
```

Put your own text in `syn_train.txt` and `syn_val.txt` (one line of text per line), then:

```
python tokenizer.py        # learns the BPE vocabulary, writes tokenizer_syn.json
python debug_overfit.py    # sanity test, needs a GPU
python train.py            # trains, writes checkpoints/ and a loss plot
python generate.py --checkpoint checkpoints/sol_best.pt --tokenizer tokenizer_syn.json
```

The training data is not included. Model size and training settings are the default arguments of `train_sol()` in `train.py`.

## What I'd do next

- Train on much more text. This is the main limit on everything above.
- Keep line breaks and conversation boundaries when loading data. Right now lines are joined with nothing between them, so the model never learns where one turn ends.
- Add learning-rate warmup and decay, and switch from Adam with L2 weight decay to AdamW.
- Tie the input embedding and output projection weights. At this vocabulary size that removes about 3.8M parameters from the largest model.
- Retrain with the final layer norm and compare against the old runs.
- Report perplexity on a standard held-out set so the models can be compared across tokenizers.
- Speed up tokenizer training, which is slow in pure Python on larger text.
