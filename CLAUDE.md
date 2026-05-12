# microgpt-oss

## What This Is

A line-by-line decomposition of [Karpathy's single-file GPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) for deep learning education. The original is 199 lines of pure Python — no dependencies — that trains and runs a GPT from scratch. We're tearing it apart to understand every piece.

## Project Structure

```
baseline/gpt.py    — The pristine original. NEVER EDIT THIS FILE.
src/               — Where decomposed code lives, moved piece by piece
tests/             — Tests that verify each piece works identically to baseline
```

## Workflow

This is a **manual, line-by-line migration**. Sean moves the code. Claude explains, documents, and verifies.

1. Sean picks a section from `baseline/gpt.py`
2. We discuss what it does — deeply, not surface level
3. Sean moves it into `src/` (new module or existing)
4. We write a test proving the moved piece behaves identically
5. Repeat until the full 199 lines live in `src/` as a proper decomposed project

## Conventions

- **Python 3.11+** — use modern syntax (match, type hints, etc.)
- **uv** for environment management
- **No deps in src/** unless explicitly discussed — the point is understanding the math, not hiding behind libraries
- **Tests compare against baseline** — if we refactor something, it must produce identical outputs given the same random seed
- **Comments are encouraged** — this is educational code, not production code

## The Modules (planned decomposition)

| File | Lines | What It Contains |
|------|-------|-----------------|
| `src/value.py` | 30-57 | `Value` class — autograd engine |
| `src/data.py` | 14-21 | Dataset loading and tokenization |
| `src/tokenizer.py` | 23-27 | Character-level tokenizer |
| `src/model.py` | 74-144 | GPT architecture (params + forward) |
| `src/train.py` | 146-184 | Training loop + Adam optimizer |
| `src/inference.py` | 186-200 | Text generation |

## Running

```bash
# Run the pristine original (proof it works)
cd baseline && python gpt.py

# Run the decomposed version (once assembled)
python -m src.main
```
