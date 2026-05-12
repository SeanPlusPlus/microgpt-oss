# microgpt-oss

> Karpathy's 199-line GPT, decomposed line-by-line for understanding.

## The Original

[This gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy is a masterpiece: a complete GPT — training and inference — in 199 lines of dependency-free Python. No PyTorch. No NumPy. Just math.

This repo tears it apart. Not to improve it — to **understand** it.

## What's Inside

| Component | What It Does |
|-----------|-------------|
| **Autograd** (Value class) | Automatic differentiation — the chain rule, implemented as a computation graph |
| **Tokenizer** | Characters → integers. The simplest possible encoding |
| **Model** | GPT-2 architecture: embeddings, multi-head attention, MLP, RMSNorm |
| **Adam** | The optimizer that updates weights based on gradients |
| **Training loop** | Forward pass → loss → backward pass → update. Repeat 1000x |
| **Inference** | Sample from the learned distribution, one token at a time |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (the fast Python package manager)

### Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter
git clone <this-repo>
cd microgpt-oss

# Create environment and install
uv sync

# Activate (optional — uv run handles this automatically)
source .venv/bin/activate
```

### Run the Baseline (Prove It Works)

```bash
cd baseline
uv run python gpt.py
```

This downloads the names dataset, trains a tiny GPT for 1000 steps (~2 min on CPU), and generates new names. You'll see loss decrease and then hallucinated names appear.

### Run Tests

```bash
uv run pytest
```

## The Decomposition Process

This isn't a refactor. It's a **study guide**.

Each piece of `baseline/gpt.py` gets:
1. Moved into its own module in `src/`
2. Annotated with explanations of the math and intuition
3. Tested against the baseline to prove correctness
4. Documented with "what would break if we changed X?"

The goal: after working through this repo, you understand every operation that happens when an LLM generates a token.

## What You'll Learn

- How automatic differentiation actually works (it's simpler than you think)
- Why attention is "all you need" (and what it's actually computing)
- How positional embeddings give the model a sense of order
- Why RMSNorm stabilizes training (and what happens without it)
- How Adam is smarter than vanilla gradient descent
- Why temperature controls "creativity" in generation

## Philosophy

The original is a poem. One file. No dependencies. This repo is the annotated edition — margin notes, cross-references, and "what if?" experiments. We acknowledge we're scribbling in the margins of art.

## License

The original code is by [@karpathy](https://github.com/karpathy). This decomposition is for educational purposes.
