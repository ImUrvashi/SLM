# SLM: a small language model from scratch

A GPT-style language model built from scratch in PyTorch and pre-trained on the cleaned GPT-4 subset of [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).

- **Model:** ~30M-parameter decoder-only Transformer (6 layers, 6 heads, 384-dim embeddings, 128-token context, GPT-2 tokenizer)
- **Training:** AdamW with warmup and cosine decay, bf16 mixed precision, gradient accumulation
- **Result so far:** validation loss 3.16 after the first run

The full pipeline (data, tokenization, model, training and text generation) is in [the notebook](notebooks/slm-with-tinystories-gpt4-clean_casual-attention.ipynb).
