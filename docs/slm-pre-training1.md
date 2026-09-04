# SLM Pre-Training 1

This document is a structured walkthrough of the notebook [slm-with-tinystories-gpt4-clean.ipynb](../notebooks/slm-with-tinystories-gpt4-clean.ipynb).

It is written to explain:

- what each step does,
- why each configuration was chosen,
- what design decisions were made,
- what issues were considered during the implementation,
- and how the full small language model, or SLM, was built from scratch and then trained.

The goal is to read like a project description of the work process itself, not just a code dump.

---
## 1. Project Goal

The notebook builds a compact GPT-style language model and pre-trains it on the cleaned GPT-4 subset of TinyStories.

The main idea is simple:

- use a clean text corpus,
- tokenize it with the GPT-2 tokenizer,
- convert token sequences into training batches,
- define a small transformer decoder,
- train with a stable optimization setup,
- and finally generate text from the trained model.

This is a classic "from data to generation" pipeline, but the notebook is careful about the practical parts that matter in real training:

- filtering noisy data,
- storing tokenized data efficiently on disk,
- using memory-mapped binary files,
- building the model in modular pieces,
- using mixed precision and gradient accumulation,
- and preserving the best checkpoint.

---

## 2. High-Level Pipeline

```text
Raw TinyStories GPT-4 subset
    |
    v
Cleaning and filtering
    |
    v
Train / Val / Test split
    |
    v
GPT-2 tokenization
    |
    v
Token IDs saved to .bin memmap files
    |
    v
Random batch sampling
    |
    v
Transformer decoder model
    |
    v
Loss computation
    |
    v
Training with AdamW + warmup + cosine decay
    |
    v
Best checkpoint saved
    |\
    | \--> Plot losses
    |
    \----> Inference / generation
```

### Why this structure matters

The notebook is not just training a model. It is solving the full systems problem of making training feasible:

- the dataset is too large to keep entirely in RAM in raw form,
- tokenization is expensive, so it is done once and cached,
- batches are sampled randomly from disk-backed token streams,
- and the model is kept intentionally small so it can train as an SLM rather than a large-scale LLM.

---

## 3. Step 1: Import the Dataset

The notebook starts by loading the cleaned TinyStories GPT-4 dataset:

- `karpathy/tinystories-gpt4-clean`

The split strategy is:

- first 10,000 rows for test,
- next 10,000 rows for validation,
- remaining rows for training.

### Why this split is reasonable

The dataset is already pre-shuffled, so taking contiguous ranges is acceptable here.

This gives:

- a fixed test set,
- a small validation set for tracking generalization,
- and the large remainder for model learning.

The notebook then wraps these splits into a `DatasetDict`, which makes downstream handling more structured and consistent.

### Design decision

This is a practical compromise between simplicity and reproducibility:

- no custom split logic is needed,
- the workflow stays close to Hugging Face conventions,
- and the train/val/test setup is easy to reason about later.

---

## 4. Cleaning Pipeline

The notebook includes a detailed markdown explanation of how the TinyStories GPT-4 subset was cleaned before training.

### 4.1 Cleaning goals

The objective was to create a corpus that is:

- mostly ASCII,
- linguistically clean,
- easy to tokenize,
- and free from formatting artifacts.

This is especially important for a character-level sanity check and for GPT-2 BPE tokenization, because noisy Unicode or markup-like symbols can introduce instability and unwanted token patterns.

### 4.2 Cleaning rules

#### Unicode normalization

The notebook standardizes punctuation and text artifacts:

- curly quotes become straight quotes,
- em and en dashes become hyphens,
- ellipses become `...`,
- stray backslashes are removed,
- repeated spaces are collapsed.

#### Non-ASCII rejection

Stories containing any non-printable ASCII characters are discarded, with newline allowed as a paragraph separator.

This decision strongly simplifies the vocabulary and reduces the chance of weird tokenization artifacts.

#### Banned character rejection

Stories containing markup-like or code-contamination symbols are removed:

```text
< > / * = _ & @ ~ # % [ ] + ( )
```

This is a careful design choice. These characters often appear in:

- HTML,
- markdown,
- code snippets,
- template artifacts,
- or prompt formatting.

For a clean tiny language model corpus, it is better to exclude them than to let them distort the distribution.

#### Minimum length filter

Stories shorter than 100 characters are removed.

This removes:

- fragments,
- empty or nearly empty entries,
- and examples too short to provide meaningful learning signal.

#### Ending punctuation check

Stories must end with one of:

```text
. ! " ?
```

This helps keep the corpus composed of complete sequences.

### 4.3 Cleaning outcome

The notebook records the rejection counts:

| Reason | Count |
| --- | ---: |
| Non-ASCII characters | 1,282 |
| Banned characters | 720 |
| Too short (< 100 characters) | 238 |
| Bad ending punctuation | 10,456 |
| Total rejected | 12,696 |

The rejection rate is only about 0.46%, which shows the GPT-4 subset is already quite clean.

### 4.4 Character inventory

After cleaning, the corpus contains 74 unique ASCII characters, including newline.

This is important because it confirms the dataset is stable and controlled before tokenization.

### Design insight

The cleaning stage is doing more than preprocessing. It is enforcing a modeling philosophy:

- fewer noisy symbols,
- fewer tokenization surprises,
- more predictable sequence boundaries,
- and a compact text distribution that is easier for a small model to learn.

---

## 5. Step 2: Tokenize the Dataset

The notebook uses `tiktoken` with the GPT-2 tokenizer.

### Why GPT-2 BPE?

GPT-2 BPE is a practical choice because:

- it is fast,
- it is well-tested,
- it produces a stable subword vocabulary,
- and it matches the standard GPT-style modeling setup.

The notebook explicitly uses:

- `enc = tiktoken.get_encoding('gpt2')`

### Processing function

The `process(example)` function:

- takes one row from the dataset,
- encodes the text using `encode_ordinary`,
- and returns:
  - `ids`: the token ID list,
  - `len`: the token sequence length.

### Why `encode_ordinary`

This avoids special-token behavior and keeps the corpus encoding focused on the raw text itself.

### Tokenization strategy

The dataset is tokenized with parallel mapping using `num_proc=8`.

This is a good design choice because tokenization is one of the slowest offline preprocessing steps, and parallelism makes the preprocessing practical on a large corpus.

### Disk-backed storage

Instead of keeping all token IDs in Python lists, the notebook concatenates them into `.bin` files using `np.memmap`.

This is one of the most important engineering decisions in the notebook.

#### Why memmap is useful

- it avoids loading the full tokenized dataset into RAM,
- it makes random batch sampling cheap later,
- it scales well to large corpora,
- and it is compatible with the streaming-like access pattern used in training.

### Token storage diagram

```text
Clean text rows
    |
    v
GPT-2 tokenizer
    |
    v
List of token IDs per story
    |
    v
Concatenation across all stories
    |
    v
train.bin / val.bin / test.bin
```

### Design decision

This is a very deliberate move away from a "load everything and batch in memory" approach.

That would be simpler for a tiny demo, but it would not scale as well and would make the notebook less robust. The memmap route is more realistic and more elegant.

---

## 6. Step 3: Create Input-Output Batches

The `get_batch(split)` function samples contiguous token blocks from the `.bin` files.

### What the batch looks like

If `block_size = T`, then:

- `x` is tokens `[i : i+T]`
- `y` is tokens `[i+1 : i+T+1]`

This creates the standard next-token-prediction training objective.

### Batch diagram

```text
Token stream:
... t0 t1 t2 t3 t4 t5 t6 t7 ...

For block_size = 4:

x = [t0 t1 t2 t3]
y = [t1 t2 t3 t4]

This teaches the model to predict the next token at every position.
```

### Why contiguous windows matter

Language modeling depends on local and medium-range context.

Using contiguous windows ensures that:

- the input context is coherent,
- the target is aligned exactly one token ahead,
- and the model learns sequential dependencies rather than shuffled token noise.

### GPU transfer choice

If CUDA is available, tensors are pinned and transferred asynchronously.

This is a performance optimization:

- pinned memory helps host-to-device transfer,
- non-blocking transfers reduce training overhead,
- and the training loop stays more efficient.

### Design decision

The notebook samples random chunks rather than fixed sequential mini-epochs.

That is useful because it:

- decorrelates batches,
- improves stochasticity,
- and avoids forcing the model through the corpus in a rigid linear order every epoch.

---

## 7. Step 4: Define the Model Architecture

This is the core modeling section. The notebook builds a GPT-style decoder-only transformer from scratch.

### 7.1 Architectural overview

```text
Token embeddings ------+
                       |
Position embeddings ---+--> Add + Dropout
                              |
                              v
                    Transformer Block 1
                              |
                              v
                    Transformer Block 2
                              |
                              v
                    ... repeated n_layer times ...
                              |
                              v
                         Final LayerNorm
                              |
                              v
                        Linear LM head
                              |
                              v
                    Logits over vocabulary
```

### 7.2 Custom LayerNorm

The notebook defines a custom `LayerNorm` wrapper rather than using PyTorch's built-in module directly.

### Why this was done

The custom version allows the bias term to be disabled when needed.

This is a small implementation detail, but it matters because:

- the model configuration can better match GPT-style design choices,
- and the normalization layer stays flexible.

### 7.3 Causal self-attention

The `CausalSelfAttention` module is the heart of autoregressive modeling.

#### What it does

- projects input embeddings into `Q`, `K`, and `V`,
- splits the channels across multiple heads,
- applies causal masking so each position can only attend to earlier tokens,
- computes attention output,
- applies output projection and dropout.

#### Key design ideas

##### Single projection for QKV

The model uses one linear layer to generate all three projections at once.

This is efficient because:

- it reduces separate matrix multiplications,
- and it follows the standard GPT implementation style.

##### Multi-head attention

Splitting the embedding dimension into multiple heads lets each head focus on different relationships in the sequence.

This gives the model richer representational power than single-head attention.

##### Flash attention fallback

The notebook checks whether `torch.nn.functional.scaled_dot_product_attention` is available.

If it is, the model uses it.
If not, it falls back to a manual masked attention implementation.

This is a strong engineering choice because it:

- uses optimized kernels when available,
- but still keeps the code portable.

### Attention matrix view

```text
For a sequence of tokens:

token 1 -> can attend to: token 1
token 2 -> can attend to: token 1, token 2
token 3 -> can attend to: token 1, token 2, token 3
...

This lower-triangular pattern is the causal mask.
```

### 7.4 MLP block

The MLP expands the embedding dimension by 4x, applies GELU, and projects back down.

### Why the 4x expansion is used

This is a standard transformer design choice because the MLP acts as the per-token nonlinear transformation block.

The transformer block is not just attention:

- attention mixes information across positions,
- the MLP transforms the representation at each position independently.

### 7.5 Transformer block

The block is:

- LayerNorm,
- self-attention,
- residual connection,
- LayerNorm,
- MLP,
- residual connection.

### Why residual connections matter

Residual paths make training deep networks more stable because gradients can flow more directly.

This is one of the main reasons transformer stacks train well at all.

### 7.6 GPTConfig

The notebook groups hyperparameters in a `GPTConfig` dataclass.

This is a good design choice because it keeps the architecture clean and explicit.

It contains:

- `block_size`
- `vocab_size`
- `n_layer`
- `n_head`
- `n_embd`
- `dropout`
- `bias`

### 7.7 Full GPT model

The model combines:

- token embeddings,
- position embeddings,
- a stack of transformer blocks,
- final LayerNorm,
- and a language modeling head.

#### Weight tying

The token embedding matrix and output projection matrix are tied.

### Why weight tying is useful

- reduces parameter count,
- can improve performance,
- and forces input and output token spaces to live in the same learned geometry.

### 7.8 Weight initialization

The notebook uses GPT-style initialization:

- normal initialization for linear and embedding weights,
- zero bias initialization,
- and a smaller initialization scale for `c_proj` weights.

### Why this matters

Transformer training can become unstable if residual branches are too large too early.

The smaller projection initialization is a practical stabilization trick.

---

## 8. Step 5: Define the Loss Function

The notebook uses cross-entropy loss over the vocabulary for next-token prediction.

### Training objective

For each token position, the model predicts the next token.

This is the standard autoregressive language modeling objective:

```text
input tokens  -> predict next tokens
```

### Why the notebook averages loss over many batches

The `estimate_loss` function evaluates both training and validation loss over multiple random batches.

This is a better signal than a single batch because it reduces noise and gives a more stable view of model progress.

### Design decision

The notebook explicitly separates:

- training mode,
- evaluation mode,
- and inference mode.

That is important because dropout and autocasting behavior must be controlled carefully.

---

## 9. Step 6 and 7: Training Configuration

This part of the notebook is where the training system is tuned carefully.

### 9.1 Core training settings

The main configuration includes:

- `learning_rate = 1e-4`
- `max_iters = 20000`
- `warmup_steps = 1000`
- `min_lr = 1e-5`
- `eval_iters = 500`
- `batch_size = 32`
- `block_size = 128`
- `gradient_accumulation_steps = 32`

### Why these values make sense

#### Learning rate

`1e-4` is a conservative and stable choice for AdamW training.

#### Warmup

Warmup prevents the optimizer from taking overly aggressive early steps before the model has settled.

#### Cosine decay

Cosine annealing gradually reduces the learning rate toward `min_lr`, which often improves convergence and final quality.

#### Larger block size

Using 128 instead of 64 allows the model to see longer context, which is important even for short story generation.

#### Batch size

Increasing batch size helps stabilize gradient estimates.

#### Gradient accumulation

Accumulating gradients simulates a larger effective batch size without requiring all samples to fit in memory at once.

### 9.2 Device setup

The notebook chooses CUDA if available, otherwise CPU.

It also sets:

- `device_type`
- `dtype`
- `ctx` as an autocast context manager

### Why mixed precision is used

Mixed precision improves training efficiency on supported GPUs.

The notebook uses:

- `bfloat16` if available,
- otherwise `float16`,
- with `GradScaler` enabled for float16.

This is a solid practical choice because it balances:

- speed,
- memory savings,
- and numerical stability.

### 9.3 Optimizer and schedule

The optimizer is `AdamW` with:

- `betas=(0.9, 0.95)`
- `weight_decay=0.1`
- `eps=1e-9`

The scheduler is a two-stage schedule:

- linear warmup,
- then cosine decay.

### Why this training stack is good

This setup reflects a careful balance:

- AdamW works well for transformers,
- warmup avoids unstable starts,
- cosine decay helps final convergence,
- weight decay regularizes the model,
- and the beta choice gives slightly smoother variance tracking.

---

## 10. Step 8: Pre-Train the SLM

The training loop is where the whole setup comes together.

### 10.1 Best checkpointing

The notebook tracks the best validation loss and saves the corresponding weights to `best_model_params.pt`.

### Why this is important

It prevents the final model from being just "the last one trained."

Instead, you keep the version that generalized best on validation.

### 10.2 Training loop structure

Each iteration does the following:

1. evaluate periodically,
2. sample a batch,
3. run forward pass under autocast,
4. scale the loss for gradient accumulation,
5. backpropagate,
6. clip gradients,
7. optimizer step,
8. scheduler step,
9. zero gradients.

### 10.3 Gradient clipping

The notebook clips gradients at `max_norm = 0.5`.

### Why clip gradients

This prevents occasional large updates from destabilizing training.

### 10.4 Why accumulation is used here

The loss is divided by `gradient_accumulation_steps` so that the effective gradient magnitude stays consistent even though updates happen less frequently.

This is a good choice when:

- memory is limited,
- batches are large,
- or you want a larger effective batch size without increasing GPU pressure too much.

### 10.5 Training behavior summary

This section shows that the notebook is not just “run a model until it works.”

It is carefully designed to manage:

- optimization stability,
- memory usage,
- evaluation timing,
- and checkpoint quality.

### Practical issues addressed

The notebook anticipates common training problems:

- mixed-precision underflow,
- unstable gradients,
- overfitting,
- noisy loss readings,
- and device placement bugs.

---

## 11. Step 9: Plot the Loss Curves

After training, the notebook plots both training loss and validation loss.

### Why this matters

Loss curves show whether training behaved as expected:

- whether loss decreases steadily,
- whether validation tracks training reasonably,
- whether the model starts overfitting,
- and whether there are signs of instability.

### Interpretation guide

```text
Train loss down, val loss down:
    good learning

Train loss down, val loss flat or up:
    possible overfitting

Both noisy or exploding:
    optimization instability
```

This step is important because the training process is not considered complete until the learning dynamics are inspected visually.

---

## 12. Step 10: Inference

The notebook reloads the best checkpoint and switches the model into evaluation mode.

### Why reload the checkpoint

This ensures that inference uses the best saved model rather than whatever weights happened to be present at the end of training.

### Generation setup

The notebook:

- encodes a prompt with the GPT-2 tokenizer,
- converts it to a tensor,
- passes it to `model.generate`,
- and decodes the generated token sequence back into text.

### Generation controls

The generation method supports:

- `temperature`,
- `top_k`,
- and context truncation to the model block size.

### Why these controls matter

#### Temperature

Controls randomness:

- lower temperature = more conservative text,
- higher temperature = more diverse text.

#### Top-k sampling

Restricts sampling to the most likely tokens, which reduces extremely unlikely outputs.

#### Context truncation

Ensures generation stays within the model's trained context window.

---

## 13. Design Decisions Summary

Here is a compact summary of the main decisions made in the notebook and why they matter.

| Decision | Why it was chosen |
| --- | --- |
| GPT-4 TinyStories subset | Cleaner and more coherent training data |
| ASCII-oriented cleaning | Reduces noise and tokenization surprises |
| GPT-2 tokenizer | Standard, fast, compatible BPE tokenizer |
| `np.memmap` token storage | Efficient large-scale disk-backed preprocessing |
| Random contiguous batch sampling | Preserves sequence structure for next-token prediction |
| Custom LayerNorm | Allows bias control and GPT-style flexibility |
| Flash-attention fallback | Uses fast kernels when available, stays portable otherwise |
| Weight tying | Saves parameters and improves output/input alignment |
| AdamW optimizer | Strong default for transformer training |
| Warmup + cosine decay | Stable early training and smoother convergence |
| Gradient accumulation | Simulates larger batch size under memory limits |
| Gradient clipping | Protects against unstable updates |
| Best-checkpoint saving | Keeps the best validation model, not just the latest model |

---

## 14. Problems Considered During the Work

Even when the notebook does not spell out every issue as an error log, the implementation clearly addresses a number of practical challenges.

### 14.1 Data cleanliness

Problem:

- noisy punctuation,
- markup-like text,
- Unicode artifacts,
- too-short examples,
- incomplete endings.

Decision:

- apply strict filtering rules.

Why:

- the model should learn story structure, not noise.

### 14.2 Memory pressure

Problem:

- tokenized corpora are large,
- loading everything into RAM is inefficient.

Decision:

- write token IDs to `.bin` memmap files.

Why:

- scalable and efficient access for training.

### 14.3 Training instability

Problem:

- large gradients,
- mixed-precision underflow,
- early-step instability.

Decision:

- use warmup,
- gradient scaling,
- gradient clipping,
- and stable optimizer settings.

### 14.4 Architectural clarity

Problem:

- transformer code can become hard to read quickly.

Decision:

- split the model into `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, and `GPTConfig`.

Why:

- improves readability,
- makes debugging easier,
- and makes the architecture explainable.

---

## 15. What This Notebook Achieves

By the end of the notebook, you have a complete small language model pipeline:

- cleaned data,
- tokenized inputs,
- disk-backed token storage,
- batch construction,
- transformer decoder model,
- training loop,
- loss tracking,
- checkpoint saving,
- and text generation.

That is a full SLM pre-training workflow, built in a way that is both understandable and reasonably efficient.
