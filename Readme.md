#  Mistral Large from Scratch (PyTorch)

> A beginner-friendly implementation of the Mistral Large architecture built completely from scratch using **PyTorch**. This project explains every major component of a modern Large Language Model (LLM) in simple words while implementing them step by step.

---

##  Overview

The goal of this project is **not to reproduce the full Mistral Large model**, which contains billions of parameters, but to understand **how modern decoder-only language models actually work internally**.

Instead of treating an LLM as a black box, this project builds each component individually and explains:

- How text becomes numbers
- How embeddings are created
- How attention works
- Why Mistral uses RoPE instead of positional embeddings
- Why RMSNorm replaces LayerNorm
- How Grouped Query Attention reduces memory
- How Sliding Window Attention improves efficiency
- Why SwiGLU performs better than ReLU
- How Transformer blocks are formed
- How language models are trained
- How text generation works using different decoding strategies

Every section contains detailed explanations so beginners can understand both **the theory** and **the implementation**.

---

# Project Architecture

```
                Input Text
                     │
                     ▼
              GPT-2 BPE Tokenizer
                     │
                     ▼
              Integer Token IDs
                     │
                     ▼
              Token Embedding Layer
                     │
                     ▼
        ┌────────────────────────────┐
        │     Transformer Block × N  │
        │                            │
        │ RMSNorm                    │
        │       │                    │
        │       ▼                    │
        │ Grouped Query Attention    │
        │   ├── RoPE                 │
        │   └── Sliding Window       │
        │       │                    │
        │ Residual Connection        │
        │       │                    │
        │ RMSNorm                    │
        │       │                    │
        │ Feed Forward Network       │
        │      (SwiGLU)              │
        │       │                    │
        │ Residual Connection        │
        └────────────────────────────┘
                     │
                     ▼
               Final Hidden States
                     │
                     ▼
              Linear Language Head
                     │
                     ▼
            Next Token Probabilities
                     │
                     ▼
           Autoregressive Generation
```

---

# Features

- Built entirely from scratch using PyTorch
- Decoder-only Transformer architecture
- GPT-2 BPE tokenizer
- Rotary Positional Embeddings (RoPE)
- RMS Normalization
- Grouped Query Attention (GQA)
- Sliding Window Attention (SWA)
- SwiGLU Feed Forward Network
- Residual Connections
- Tiny Shakespeare training pipeline
- Training and validation loss plotting
- Greedy decoding
- Temperature sampling
- Top-P (Nucleus) sampling
- Save and reload trained models

---

# Technologies Used

- Python
- PyTorch
- tiktoken
- Matplotlib
- NumPy

---

# Model Components

## 1. Tokenization

The model first converts text into integer token IDs using the GPT-2 Byte Pair Encoding (BPE) tokenizer.

Example:

```
Input:
I love pizza

↓

Tokens:

[40, 1842, 22914]
```

The tokenizer converts words (or parts of words) into numerical IDs that the neural network can understand.

---

## 2. Token Embeddings

Token IDs themselves have no meaning.

An embedding layer converts every token into a dense vector.

Example

```
Token ID

↓

Embedding Vector

↓

[0.24, -1.82, 0.91, ...]
```

These vectors contain semantic information that the model learns during training.

---

## 3. RMSNorm

Instead of Layer Normalization, Mistral uses **Root Mean Square Normalization (RMSNorm)**.

Advantages:

- Faster
- Requires fewer computations
- More memory efficient
- Stable training

RMSNorm normalizes the magnitude of vectors without centering their mean.

---

## 4. Rotary Positional Embedding (RoPE)

Transformers must know the position of every token.

Instead of adding positional embeddings, Mistral rotates the Query and Key vectors.

Advantages:

- Better handling of long contexts
- Relative positional understanding
- Better extrapolation to unseen sequence lengths

---

## 5. Grouped Query Attention (GQA)

Traditional Multi-Head Attention creates separate Query, Key and Value heads for every attention head.

Mistral instead shares Key and Value heads across multiple Query heads.

Example

```
32 Query Heads

↓

8 Key Heads

↓

8 Value Heads
```

Benefits:

- Smaller KV cache
- Less memory usage
- Faster inference
- Almost identical performance

---

## 6. Sliding Window Attention

Standard Transformers attend to every previous token.

Complexity:

```
O(n²)
```

Mistral instead only attends to the most recent tokens inside a fixed window.

Example

```
Current token

↓

Looks only at previous 4096 tokens
```

Benefits:

- Much lower memory usage
- Faster inference
- Supports long documents efficiently

---

## 7. Feed Forward Network (SwiGLU)

After attention, every token passes through a Feed Forward Network.

Instead of ReLU or GELU, Mistral uses **SwiGLU**.

Pipeline

```
Hidden State

↓

Up Projection

↓

Gate Projection

↓

SiLU Activation

↓

Element-wise Multiplication

↓

Down Projection
```

Benefits

- Better gradient flow
- Higher accuracy
- Better representation learning

---

## 8. Transformer Block

One Transformer block contains

```
Input

↓

RMSNorm

↓

Grouped Query Attention

↓

Residual Connection

↓

RMSNorm

↓

Feed Forward Network (SwiGLU)

↓

Residual Connection

↓

Output
```

Multiple Transformer blocks are stacked together to form the language model.

---

## 9. Language Modeling Head

The final hidden vectors are projected back into vocabulary space.

The model predicts the probability of every possible next token.

Example

```
"The cat sat on the"

↓

mat     72%

floor   12%

chair    8%

tree     2%
```

The next token is then selected using a decoding strategy.

---

# Training Pipeline

The notebook trains the model on the Tiny Shakespeare dataset.

Training steps:

1. Download dataset
2. Tokenize text
3. Split into train and validation sets
4. Create random batches
5. Forward pass
6. Compute Cross Entropy Loss
7. Backpropagation
8. Gradient clipping
9. Optimizer update
10. Learning rate scheduling
11. Validation evaluation
12. Save model

---

# Training Flow

```
Raw Text

↓

Tokenizer

↓

Token IDs

↓

Mini Batches

↓

Forward Pass

↓

Loss Calculation

↓

Backpropagation

↓

Optimizer Step

↓

Repeat
```

---

# Loss Visualization

The notebook plots:

- Training loss
- Validation loss

These graphs help determine whether:

- the model is learning correctly
- training is stable
- overfitting is occurring

---

# Text Generation

After training, the notebook demonstrates three decoding methods.

---

## Greedy Decoding

Always selects the token with the highest probability.

Pros

- Fast
- Deterministic

Cons

- Can become repetitive

---

## Temperature Sampling

Adjusts randomness before sampling.

Low Temperature

```
0.5
```

Produces

- safer
- more focused
- more predictable text

High Temperature

```
1.4
```

Produces

- creative
- diverse
- less predictable text

---

## Top-P (Nucleus) Sampling

Instead of considering every possible token, only the smallest set of tokens whose cumulative probability exceeds **P** is sampled.

Example

```
Top P = 0.9

↓

Only the most probable tokens whose total probability reaches 90% are considered.
```

This creates more natural and coherent text.

---

# Saving the Model

After training, the notebook stores:

- learned weights
- model configuration

This allows the model to be reloaded later without retraining.

---

# File Structure

```
MISTRAL_LARGE_FROM_SCRATCH.ipynb
│
├── Model Configuration
├── Tokenizer
├── RMSNorm
├── Rotary Positional Embedding
├── Grouped Query Attention
├── Sliding Window Attention
├── SwiGLU Feed Forward Network
├── Transformer Block
├── Complete Mistral Model
├── Dataset Preparation
├── Training Loop
├── Evaluation
├── Loss Visualization
├── Text Generation
└── Save / Load Model
```

---

# What This Project Demonstrates

This implementation covers the core ideas behind modern decoder-only language models, including:

- Transformer architecture
- Token embeddings
- Decoder-only language modeling
- RMS Normalization
- Rotary Positional Embeddings
- Grouped Query Attention
- Sliding Window Attention
- SwiGLU activation
- Residual connections
- Cross entropy training
- AdamW optimization
- Gradient clipping
- Learning rate scheduling
- Autoregressive text generation
- Greedy decoding
- Temperature sampling
- Top-P sampling

---

# Limitations

This project is designed for **learning and experimentation**, not production use.

Compared to the real Mistral Large model:

- Uses a much smaller architecture
- Trains on Tiny Shakespeare instead of trillions of tokens
- Does not use Mixture of Experts (MoE)
- Does not include distributed training
- Does not implement Flash Attention
- Does not support large-scale inference optimizations

Despite these simplifications, the implementation closely follows the architecture and design principles of modern decoder-only Transformers, making it an excellent educational resource.

---

# Learning Outcomes

By completing this project, you will understand:

- How Large Language Models process text
- Why Mistral is more efficient than standard Transformers
- How attention mechanisms work internally
- How modern positional embeddings function
- How decoder-only language models are trained
- How next-token prediction works
- How text generation algorithms differ
- How individual Transformer components combine to form an end-to-end language model

---

# Future Improvements

Possible extensions to this project include:

- Fine-tuning using LoRA or QLoRA
- Support for instruction-tuning datasets such as Alpaca
- Mixed precision (FP16/BF16) training
- Flash Attention
- KV Cache optimization
- Multi-GPU training
- Hugging Face model export
- Larger datasets and vocabulary
- Evaluation using perplexity and benchmark datasets

---

# Acknowledgements

This project was created as an educational implementation to understand the internal architecture of **Mistral Large** and modern decoder-only Transformer models. It is inspired by the design principles described in the Mistral architecture and implemented from first principles using PyTorch for learning purposes.
