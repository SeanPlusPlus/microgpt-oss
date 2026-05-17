# Decomposition Progress

Baseline: `baseline/gpt.py` (199 lines, never edit)

## Completed

### `src/data.py` (baseline lines 14–21)
- Downloads `names.txt` from Karpathy's makemore repo if not present
- Reads all names, strips whitespace, shuffles with seed 42
- Returns `list[str]` of ~32k names
- Deterministic order thanks to `random.seed(42)` at module level

### `src/tokenizer.py` (baseline lines 23–27)
- Builds a character-level vocabulary from the dataset
- `uchars`: sorted unique characters → their index is their token ID (e.g. `a`→0, `b`→1)
- `BOS`: beginning/end of sequence token, gets the ID after all chars (~27)
- `vocab_size`: total tokens = chars + 1 (for BOS)
- Returns a tuple: `(uchars, BOS, vocab_size)`

### `src/value.py` (baseline lines 30–57) — IN PROGRESS
- The autograd engine — a scalar computation graph with automatic differentiation
- Each `Value` node stores: its data, its gradient, its parent nodes, and local partial derivatives
- Core ops (`+`, `*`, `**`, `log`, `exp`, `relu`) build the graph during forward pass
- Convenience ops (`-`, `/`, negation, reflected ops) delegate to core ops
- **Missing:** `backward()` method (lines 58–72) — topological sort + chain rule propagation

## Remaining

| File | Baseline Lines | What It Contains |
|------|---------------|-----------------|
| `src/value.py` | 58–72 | `backward()` — topo sort the graph, propagate gradients via chain rule |
| `src/model.py` | 74–144 | GPT architecture: embedding, attention, MLP, layer norm, full forward pass |
| `src/train.py` | 146–184 | Training loop: batch construction, loss calculation, Adam optimizer |
| `src/inference.py` | 186–200 | Text generation: autoregressive sampling with temperature + KV cache |
| `src/main.py` | — | Entrypoint that wires all modules together |

## Next Step

Add `backward()` to `Value`, then write a test proving the autograd engine computes correct gradients.
