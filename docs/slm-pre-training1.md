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

The dataset is already pre-shuffled(order of stories made randomised instead of 1, 2, 3..), so taking contiguous ranges is acceptable here.

This gives:

- a fixed test set,
- a small validation set for tracking generalization,
- and the large remainder for model learning.

The notebook then wraps these splits into a `DatasetDict`, which makes downstream-handling(Grouping the splits makes the data easier to work with in later steps) more structured and consistent.

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

#### Embedding dimensions and tensor shapes

Our configuration uses `vocab_size = 50257`, `block_size = 128`, and `n_embd = 384`. Here, `block_size` is the maximum context length, and `n_embd` is the number of learned values representing each token.

| Matrix or tensor | Shape | Meaning |
| --- | --- | --- |
| Token embedding weights (`wte.weight`) | `[50257, 384]` | One learned 384-dimensional vector for every vocabulary token |
| Position embedding weights (`wpe.weight`) | `[128, 384]` | One learned 384-dimensional vector for each supported position |
| Input token IDs for one full-length sequence | `[128]` | The integer IDs of the 128 input tokens |
| Token embeddings for one full-length sequence | `[128, 384]` | The vectors looked up for those 128 token IDs |
| Input token IDs for a batch of 32 full-length sequences | `[32, 128]` | 32 sequences, each containing 128 token IDs |
| Token embeddings for that batch | `[32, 128, 384]` | One 384-dimensional vector per token in each sequence |
| Combined token and position embeddings for that batch | `[32, 128, 384]` | The input representations passed through dropout and into the Transformer blocks |

The **full token embedding matrix** has shape `[50257, 384]`. Looking up 128 token IDs selects 128 rows from that matrix, producing an **embedded input sequence** of shape `[128, 384]`. Repeated token IDs look up the same row.

```text
Input token IDs:             [32, 128]
                                  |
                         Token embedding lookup
                                  |
                                  v
Token embeddings:           [32, 128, 384]
Position embeddings:            [128, 384]
                                  |
                      Add (broadcast across batch)
                                  |
                                  v
Combined embeddings:        [32, 128, 384]
```

Both embedding matrices are trainable: token embeddings contain **50,257 × 384 = 19,298,688 parameters**, and position embeddings contain **128 × 384 = 49,152 parameters**. The embedded input tensors are lookup results; they do not introduce additional model parameters.

More generally, for batch size `B` and sequence length `T`, input IDs have shape `[B, T]` and embedded inputs have shape `[B, T, 384]`, where `T <= 128` in our model. The full embedding weight matrices keep their shapes regardless of the batch size or input length.

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

---

## 16. Q&A

### Q1. What is the size of our model and dataset?

Our model has **29,995,392 trainable parameters**, or approximately **30 million parameters**. Its architecture uses:

- 6 Transformer layers,
- 6 attention heads per layer,
- 384-dimensional embeddings,
- a 128-token context window,
- and a GPT-2 vocabulary of 50,257 tokens.

The input token embeddings and output projection share weights, so those parameters are counted only once. The saved `best_model_params.pt` checkpoint occupies approximately **120 MB** on disk.

The dataset is **TinyStories GPT-4 Clean** (`karpathy/tinystories-gpt4-clean`). The notebook records the following story counts, and the local `.bin` file sizes give the token counts:

| Split | Stories | Tokens |
| --- | ---: | ---: |
| Training | 2,712,634 | 535,627,045 |
| Validation | 10,000 | 1,979,438 |
| Test | 10,000 | 1,989,889 |
| **Total** | **2,732,634** | **539,596,372** |

The model therefore has about **30 million parameters**, with approximately **536 million tokens available for training**. This is the size of the training corpus; the number of tokens actually processed depends on the training run.

The three tokenized files together occupy approximately **1.08 GB** on disk. Each token ID takes 2 bytes because the files use `uint16`. These storage figures describe the tokenized dataset, rather than the original text or download size.

### Q2. Why do we store all the tokens in `.bin` files? What happens with and without them?

The `.bin` files are a **cached numerical version of our stories**. The model takes token IDs as input, so we tokenize the text once and save those IDs for reuse.

```text
Story text -> GPT-2 tokenizer -> token IDs -> train.bin / val.bin / test.bin
                                                        |
                                                        v
                                                Training batches
```

Each file contains a flat sequence of token IDs stored as `uint16` integers. This works because GPT-2's largest token ID, 50,256, fits within the unsigned 16-bit range of 0 to 65,535.

| With `.bin` files | Without `.bin` files |
| --- | --- |
| Tokenize once and reuse the results across training runs. | Supply tokens another way, such as tokenizing during training or using a different cache. |
| Read random 128-token windows directly from the saved token stream. | Fetch stories, tokenize them, and assemble training windows through another data pipeline. |
| Use memory mapping to access slices without loading the whole dataset into RAM. | Keeping all tokens in RAM requires more memory; streaming avoids that but needs another input pipeline. |
| Restart training without repeating tokenization. | May need to repeat tokenization after restarting if no reusable token cache exists. |

In our notebook, preprocessing creates the token files when `train.bin` is absent. During training, `get_batch()` opens `train.bin` or `val.bin` using `np.memmap` and samples contiguous windows. If we skip preprocessing and start training without those files, batch loading fails because it expects them to exist.

The current preprocessing guard checks only whether `train.bin` exists, so an existing `train.bin` does not guarantee that `val.bin` and `test.bin` are also present.

The files themselves do not improve learning quality. Supplying the same token sequences through another pipeline can provide the same training data. Their benefit is avoiding repeated tokenization, making restarts convenient, and keeping RAM usage manageable.

After training, **text generation does not need the dataset `.bin` files**. It needs the saved model weights, the matching model configuration, and the tokenizer.

### Q3. How many neurons and parameters does our feedforward network have, and why do we use it?

Each Transformer block contains a feedforward neural network (FFN), implemented as the `MLP` class in our notebook:

```text
384 input features -> Linear -> 1,536 hidden neurons -> GELU -> Linear -> 384 output features -> Dropout
```

The hidden width is **4 × n_embd = 4 × 384 = 1,536**. With `bias=True`, the parameter counts are:

| Layer | Weights | Biases | Total parameters |
| --- | ---: | ---: | ---: |
| First linear: 384 -> 1,536 | 384 × 1,536 = 589,824 | 1,536 | 591,360 |
| Second linear: 1,536 -> 384 | 1,536 × 384 = 589,824 | 384 | 590,208 |
| **One FFN** | **1,179,648** | **1,920** | **1,181,568** |

GELU and dropout have no trainable parameters. Across our **6 Transformer blocks**, the FFNs contain **6 × 1,181,568 = 7,089,408 parameters**, or approximately **7.09 million**. Each block has its own FFN weights; within a block, the same weights are applied independently to every token position.

#### Why use a feedforward network after attention?

Attention lets a token gather information from other allowed positions. The FFN then transforms the resulting features at that token's position:

- **Learn feature combinations:** the learned linear layers mix the 384 features, while GELU introduces nonlinearity so the FFN can learn transformations that a single linear layer cannot represent.
- **Provide more intermediate capacity:** expanding to 1,536 hidden features gives the network more room to represent useful feature combinations before projecting back to 384. The 4× expansion is a common design choice, rather than a mathematical requirement.
- **Keep the residual connection compatible:** projecting back to 384 makes the FFN output match the input shape, allowing the block to add them using `x = x + self.mlp(self.ln2(x))`.

For a full-length input batch, the FFN shapes are:

```text
[B, 128, 384] -> [B, 128, 1536] -> [B, 128, 384]
```

The FFN mixes features within each token representation; attention mixes information across token positions. Increasing the batch size or sequence length increases computation and activation memory, but **does not increase the FFN's parameter count**.

### Q4. How many trainable parameters are in each Transformer block? Do they all come from attention and the FFN?

Each of our Transformer blocks has **1,774,464 trainable parameters**, or approximately **1.77 million**. Most come from attention and the FFN; the block's two LayerNorms also contribute trainable parameters.

Our block applies these components in this order:

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

The calculations below use `n_embd = 384`, `n_head = 6`, and `bias = True`.

#### How to count parameters in a linear layer

For `nn.Linear(input_features, output_features, bias=True)`:

```text
Weight parameters = input_features × output_features
Bias parameters   = output_features
Total parameters  = input_features × output_features + output_features
```

Each output neuron has one weight for every input feature and one bias. In PyTorch, the stored weight tensor has shape `[output_features, input_features]`; either dimension order gives the same parameter count.

#### 1. Causal self-attention: 591,360 parameters

The attention module has two learned linear projections.

The first, `c_attn`, projects each 384-dimensional token representation into queries, keys, and values together. Each needs 384 dimensions, so the combined output width is **3 × 384 = 1,152**:

```text
c_attn = Linear(384, 1152)
Weights: 384 × 1,152 = 442,368
Biases:                 1,152
Total:                443,520
```

Queries, keys, and values are each split into **6 heads of 64 dimensions**, since `384 / 6 = 64`. Splitting and reshaping introduce no new parameters. The combined projection already includes all heads, so we do not multiply its parameter count by 6.

After attention, the head outputs are concatenated back into 384 features. The output projection, `c_proj`, learns how to mix these features:

```text
c_proj = Linear(384, 384)
Weights: 384 × 384 = 147,456
Biases:                 384
Total:              147,840

Attention total = 443,520 + 147,840 = 591,360
```

The query-key products, softmax, causal mask, weighted sum of values, and dropout do not introduce trainable parameters. Attention scores are calculated from the current input; they are not a stored parameter matrix.

#### 2. Feedforward network: 1,181,568 parameters

As described in Q3, the FFN expands the feature width from 384 to 1,536, applies GELU, and projects back to 384:

```text
First linear:  (384 × 1,536) + 1,536 = 591,360
Second linear: (1,536 × 384) + 384   = 590,208

FFN total = 591,360 + 590,208 = 1,181,568
```

GELU and dropout add no trainable parameters.

#### 3. Two LayerNorms: 1,536 parameters

Our LayerNorm normalizes across the 384 features of each token. Each LayerNorm learns:

- **384 scale values** (`weight`, often called gamma),
- **384 bias values** (`bias`, often called beta).

The mean and variance are calculated from the input and are not trainable parameters.

```text
One LayerNorm: 384 + 384 = 768
Two LayerNorms: 2 × 768 = 1,536
```

The first LayerNorm is applied before attention, and the second before the FFN.

#### 4. Add the components for one block

| Component | Trainable parameters |
| --- | ---: |
| Causal self-attention | 591,360 |
| Feedforward network | 1,181,568 |
| LayerNorm before attention | 768 |
| LayerNorm before FFN | 768 |
| **Total per Transformer block** | **1,774,464** |

Residual additions have no trainable parameters. Batch size and context length change how often these weights are used and how many activations are computed; they do not multiply the number of parameters in a block.

#### 5. Extend the count to all six blocks and the full model

Each block has its own weights, so all six blocks together contain:

```text
6 × 1,774,464 = 10,646,784 parameters
```

Token embeddings, position embeddings, and the final LayerNorm sit outside these blocks. Including them gives the complete model count:

| Model component | Calculation | Trainable parameters |
| --- | --- | ---: |
| Token embeddings | 50,257 × 384 | 19,298,688 |
| Position embeddings | 128 × 384 | 49,152 |
| Six Transformer blocks | 6 × 1,774,464 | 10,646,784 |
| Final LayerNorm | 384 + 384 | 768 |
| LM head (shared token embedding weights, no bias) | No additional parameters | 0 |
| **Full model** | | **29,995,392** |

The LM head uses the same trainable weight matrix as the token embeddings. Its zero additional count means those shared parameters have already been counted, rather than that the head has no learned weights.

### Q5. What is weight tying, why do we use it, and are we sharing `x` or the embedding weights?

**Weight tying shares the learned weight matrix between the input token embedding layer and the output language modeling head.** We share the weights, rather than the input or hidden representations called `x`.

Our notebook ties the weights with:

```python
self.transformer.wte.weight = self.lm_head.weight
```

After this assignment, both layers reference the **same trainable parameter**, rather than independent copies. Call this shared matrix `W`; its shape is `[50257, 384]`.

#### How the input embedding uses the shared matrix

A token ID is an integer identifying a vocabulary entry. The embedding layer uses that ID to select a row of `W`:

```python
tok_emb = self.transformer.wte(idx)
```

For example, token ID 42 selects `W[42]`, a vector of 384 learned values. The token ID acts as an index; we do not multiply the integer 42 by the matrix. The numbers assigned to token IDs do not themselves encode semantic distances between tokens.

```text
Input IDs:           [B, T]
Embedding weights W: [50257, 384]
Lookup result:       [B, T, 384]
```

These vectors receive position embeddings and pass through the six Transformer blocks and final LayerNorm. The resulting `x` contains the features calculated for the current input.

#### How the output head reuses the shared matrix

The output head converts the final 384-dimensional representation at each selected position into one score for every vocabulary token:

```python
logits = self.lm_head(x)
# Equivalent calculation, because the head has no bias:
logits = x @ W.T
```

```text
Final representation x: [B, T, 384]
Shared matrix W.T:      [384, 50257]
Vocabulary logits:      [B, T, 50257]
```

For vocabulary token `j`, its logit is the dot product between the final hidden representation and row `W[j]`. If we call the final representation at one position `h`, this is `logit_j = h @ W[j]`. Here, `h` names a computed representation; it is not another embedding weight or something used in the initial lookup.

During training, the model calculates logits at every input position. During generation, our code selects only the last position before applying the head, producing logits of shape `[B, 1, 50257]`. Softmax and token selection happen after the head; they do not change which weights are shared.

#### Weights versus computed representations

| Item | Meaning | Behavior |
| --- | --- | --- |
| `W` | Learned token embedding and output projection weights | Shared by the two layers; updated during training and saved in the checkpoint |
| `tok_emb` | Token vectors retrieved for the current input IDs | Calculated by looking up rows of `W` |
| `x` / `h` | Intermediate or final hidden representations | Calculated from the current input; not shared trainable parameters |

The word "shared" refers to the two layers referencing `W`. It does not mean that the input embedding vectors and final hidden representations have identical values. Gradients from both uses contribute to training the same matrix.

#### Why use weight tying?

The shared matrix contains:

```text
50,257 × 384 = 19,298,688 parameters
```

Without tying, the input embedding and output head would each have an independent matrix of this size. With the rest of our architecture unchanged:

| Configuration | Total model parameters |
| --- | ---: |
| Without weight tying | 49,294,080 |
| With weight tying | 29,995,392 |
| **Parameters saved** | **19,298,688** |

Sharing therefore reduces parameter storage, gradient memory, and optimizer-state memory. The full vocabulary projection still needs to be computed; weight tying does not remove that step or reduce its output size.

It also links how tokens are represented on input with how they are scored on output. This constraint can act as regularization and help generalization, although improved accuracy is not guaranteed. The tradeoff is less flexibility than using two independent matrices.

### Q6. What does temperature do to logits, and why do we apply it during generation?

**Temperature controls how concentrated or spread out the next-token probabilities are.** In `generate()`, we select the final position's vocabulary logits and divide them by temperature before softmax:

```python
logits = logits[:, -1, :] / temperature
```

For vocabulary token `i`, with original logit `z_i` and positive temperature `T`, the resulting probability before any top-k filtering is:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Here, `T` means temperature; it is separate from the sequence-length notation used in earlier shape examples.

| Temperature | Effect on probabilities | Effect when sampling |
| --- | --- | --- |
| `0 < T < 1` | Sharpens the distribution | Favors the highest-scoring tokens more strongly |
| `T = 1` | Leaves the logits unchanged | Uses the model's original distribution |
| `T > 1` | Flattens the distribution | Gives lower-scoring tokens more opportunity |

For example, suppose three tokens have logits `[2, 1, 0]`, with no top-k filtering:

| Temperature | Probability of first token | Probability of second token | Probability of third token |
| --- | ---: | ---: | ---: |
| `0.5` | 86.7% | 11.7% | 1.6% |
| `1.0` | 66.5% | 24.5% | 9.0% |
| `2.0` | 50.6% | 30.7% | 18.6% |

Dividing by a positive temperature preserves the ranking of the logits. The highest-scoring token remains highest, but its probability advantage changes. Our code samples using `torch.multinomial()`, so these probability changes affect which token can be selected on a given draw. If we always selected `argmax`, positive temperature scaling would not change the selected maximum.

#### Why use temperature here rather than throughout the model?

During training, we pass the original vocabulary logits to cross-entropy loss:

```python
loss = F.cross_entropy(logits, targets)
```

Cross-entropy handles log-softmax internally. Our training uses no additional temperature scaling, equivalent to temperature 1. Applying a different temperature during training would change the loss and gradients, making it a separate training-design choice.

During generation, temperature lets us adjust sampling behavior without retraining or changing the learned weights. We apply it immediately before the vocabulary softmax because that is the distribution we want to control. Scaling embeddings or hidden features would instead change intermediate model computations.

Attention also divides its scores by the square root of the head dimension, `sqrt(64)`. That scaling keeps attention scores well behaved; it is separate from our adjustable generation temperature.

Temperature must be positive in this implementation. Setting it to zero would divide by zero; use explicit `argmax` selection when greedy decoding is desired.

### Q7. What is top-k sampling, why do we use it, and how does it work with temperature?

**Top-k sampling restricts the next-token choice to the highest-scoring vocabulary candidates.** It prevents low-scoring tokens outside the retained set from being sampled while preserving variety among eligible tokens.

For example, with `top_k = 3`:

| Candidate token | Logit before filtering | Logit after filtering |
| --- | ---: | ---: |
| cat | 5.0 | 5.0 |
| dog | 4.0 | 4.0 |
| bird | 3.0 | 3.0 |
| car | 2.0 | Negative infinity |
| table | 1.0 | Negative infinity |

Softmax gives the excluded tokens probability zero and normalizes probabilities over the retained candidates.

Our implementation is:

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Line by line:

1. `if top_k is not None` enables filtering only when a value is provided.
2. `min(top_k, logits.size(-1))` caps the requested number at the vocabulary size. When supplied, `top_k` should be a positive integer.
3. `torch.topk(...)` returns the highest scores and their vocabulary indices for each batch row. We keep the scores as `v` and discard the indices using `_`.
4. `v[:, [-1]]` selects the lowest score within each row's top-k results, keeping shape `[B, 1]` so the threshold broadcasts across vocabulary scores.
5. Scores below that threshold become negative infinity, giving them zero probability after softmax.
6. `F.softmax(...)` calculates probabilities over the surviving candidates.
7. `torch.multinomial(..., num_samples=1)` samples one next-token ID per batch row according to those probabilities.

Because our code filters by score threshold, ties at the boundary can retain more than `k` tokens. With a unique maximum, `top_k = 1` selects the same token as greedy decoding; tied maxima can still be sampled in this implementation.

| Setting | Behavior |
| --- | --- |
| Small positive `k` | Restricts sampling to fewer high-scoring candidates |
| Large `k` | Allows a broader set of candidates |
| `top_k = None` | Skips top-k filtering |

Top-k does not guarantee that the generated text is correct or coherent. It controls candidate eligibility based on the model's current scores.

#### Temperature and top-k together

Our generation order is:

```text
Vocabulary logits -> Divide by temperature -> Top-k filtering -> Softmax -> Sample next token
```

**Top-k determines which tokens remain eligible; temperature controls their relative probabilities.** Positive temperature scaling preserves score rankings, so it does not change the eligible top-k set mathematically. Both settings affect generation without changing the model's learned parameters.

### Q8. What is weight decay, why do we use it, and what happens without it?

**Weight decay gently shrinks parameters during optimizer updates.** It provides regularization: it discourages unnecessarily large parameter values and can help reduce overfitting.

Our optimizer is configured with:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-9,
)
```

In addition to its adaptive gradient update, AdamW applies a separate decay operation:

$$
w \leftarrow (1 - \eta\lambda)w
$$

Here, `w` is a parameter, `eta` is the current learning rate, and `lambda` is the weight decay coefficient. The full optimizer update also includes the gradient-based change; the equation above shows only the decay component. AdamW keeps decay separate from its gradient momentum and variance calculations. See the [PyTorch AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

At learning rate `1e-4` and weight decay `0.1`:

```text
Decay factor = 1 - (0.0001 × 0.1) = 0.99999
A weight of 2.0 becomes 1.99998 from the decay operation alone.
```

So **`weight_decay=0.1` does not mean shrinking weights by 10% per update**. The shrinkage depends on the current learning rate too. A weight can still increase overall if the gradient-based update outweighs the shrinkage.

Our code passes all model parameters in one optimizer group, so it does not exclude biases or LayerNorm parameters from decay. Decay is applied at optimizer updates, rather than at every accumulated small batch.

**Without weight decay:** training still works, but this regularization is absent. The model may overfit more, although removing excessive decay can improve results. The useful strength depends on the model, data, and training budget.

Weight decay and learning-rate decay are different: weight decay shrinks parameters, while learning-rate decay reduces the scale of optimizer updates.

### Q9. What are learning-rate decay, schedulers, `LinearLR`, `CosineAnnealingLR`, and `SequentialLR`?

**The learning rate controls how strongly the optimizer changes parameters.** For a simple gradient-descent update:

$$
w \leftarrow w - \eta\nabla_w L
$$

AdamW uses an adaptive update direction, but the learning rate still scales its update. Larger rates allow larger changes; smaller rates allow finer adjustments.

#### Why decay the learning rate?

A rate that supports useful progress early in training can keep updates too large later. Decreasing it can help the model settle into a lower-loss region with smaller adjustments.

Our intended schedule is:

```text
Small starting rate -> Linear warmup -> Peak rate -> Cosine decay -> Small final rate
       1e-7                               1e-4                          1e-5
```

These are intended rate values; Q11 explains why the current run's step allocation does not provide a sustained decay phase.

**Without learning-rate decay:** a constant rate can still work. If it stays too large, loss may fluctuate instead of settling; if it is small throughout, initial progress may be slow. Decay is a useful training choice rather than a requirement.

#### What is a scheduler?

A scheduler changes the optimizer's learning rate according to a rule. It does not add model parameters or perform backpropagation.

```python
scheduler.step()
```

Our loop calls this after each optimizer update. Therefore, a scheduler "step" here means an optimizer-update interval, rather than a small-batch iteration or a full pass over the dataset.

**Without a scheduler:** the rate remains fixed unless we change it manually. Different scheduling rules can also replace the ones below.

#### `LinearLR`: increase the rate during warmup

```python
scheduler_warmup = LinearLR(
    optimizer,
    start_factor=1e-3,
    total_iters=warmup_optimizer_steps,
)
```

`LinearLR` changes a multiplier on the base rate linearly. We use its default `end_factor=1.0`, so the multiplier moves from `0.001` toward `1.0`:

```text
Base learning rate = 1e-4
Starting rate      = 1e-4 × 1e-3 = 1e-7
Warmup target      = 1e-4 × 1.0  = 1e-4
```

Warmup avoids large updates immediately after initialization, while the model and Adam's running gradient statistics are still getting established. See the [PyTorch LinearLR documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html).

**Without warmup:** training starts at the full configured rate. This may work, but an aggressive starting rate can cause unstable early updates.

#### `CosineAnnealingLR`: decrease the rate after warmup

```python
scheduler_decay = CosineAnnealingLR(
    optimizer,
    T_max=max(1, optimizer_steps - warmup_optimizer_steps),
    eta_min=min_lr,
)
```

Over its configured decay duration, the learning rate follows a cosine curve from the peak rate toward `eta_min`:

$$
\eta_t = \eta_{\min}
+ \frac{\eta_{\max}-\eta_{\min}}{2}
\left(1+\cos\left(\frac{\pi t}{D}\right)\right)
$$

Here, `t` counts scheduler steps within the decay phase, and `D` is `T_max`, the duration of that phase. Our intended peak is `1e-4`, and `eta_min = min_lr = 1e-5`. This scheduler does not automatically restart the rate. See the [PyTorch CosineAnnealingLR documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).

**Without cosine decay:** we can use another decay rule, manually adjust the rate, or keep it constant. Cosine is a smooth scheduling choice, rather than the only valid one.

#### `SequentialLR`: switch from warmup to decay

```python
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_optimizer_steps],
)
```

This runs the warmup scheduler first, then switches to the cosine scheduler at the milestone. It coordinates the two phases. See the [PyTorch SequentialLR documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html).

**Without `SequentialLR`:** we would need to implement that switch ourselves. Stepping both schedulers throughout the run would not reproduce the intended sequence.

### Q10. What is gradient accumulation, why do we use it, and what happens without it?

**Gradient accumulation combines gradients from several small batches before updating the parameters once.** This allows a larger effective batch without keeping all its activations in GPU memory simultaneously.

Our settings are:

```python
batch_size = 32
block_size = 128
gradient_accumulation_steps = 32
```

Each small batch, also called a microbatch, contains 32 sequences of 128 tokens. We process 32 such batches before a normal optimizer update:

```text
Microbatch 1  -> Forward + backward -> Retain gradients
Microbatch 2  -> Forward + backward -> Add to gradients
...
Microbatch 32 -> Forward + backward -> Add to gradients
              -> Clip accumulated gradients
              -> Update weights once
              -> Clear gradients
              -> Advance the learning-rate scheduler
```

The effective batch per update is:

```text
32 sequences per microbatch × 32 microbatches = 1,024 sequences
1,024 sequences × 128 positions = 131,072 token positions
```

These are sampled training positions; they are not necessarily distinct tokens or complete stories from the corpus. See the [PyTorch gradient accumulation example](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation).

#### Why divide the loss?

```python
loss = loss / gradient_accumulation_steps
scaler.scale(loss).backward()
```

PyTorch adds gradients to existing `.grad` values on successive backward passes. Dividing each microbatch's mean loss by 32 makes the accumulated gradient represent the average across 32 equal-sized microbatches:

$$
g_{\mathrm{accumulated}} = \frac{1}{32}\sum_{m=1}^{32}\nabla L_m
$$

Without the division, the accumulated gradient would be the sum, with a different scale. Gradient clipping and Adam's adaptation mean the final parameter change would not simply become exactly 32 times larger, but the intended averaged-gradient calculation would be lost.

We do not clear gradients between microbatches. After accumulation, our code unscales and clips them, performs the optimizer update, and clears them before the next group. The model parameters remain unchanged while gradients for a group are being collected.

#### What does accumulation save, and what does it cost?

We retain parameter-gradient buffers across microbatches, while each microbatch's computation graph can be released after backward. This avoids retaining activations for all 1,024 sequences at once. It still requires the forward and backward computations for every microbatch; accumulation does not remove that work or guarantee the same speed as one large physical batch.

For equal-sized microbatches, averaging their gradients approximates a single large-batch calculation at the same parameter values. Floating-point arithmetic and dropout can prevent bit-for-bit identical results.

**Without accumulation**, there are two main choices:

| Choice | Consequence |
| --- | --- |
| Keep the physical batch at 32 and update every batch | Smaller, generally noisier gradient estimates and more frequent optimizer updates |
| Increase the physical batch to 1,024 | Similar effective batch size, but much larger activation-memory requirements and possible GPU out-of-memory errors |

Neither choice automatically makes training fail. Changing the accumulation factor changes the update count and may require retuning the learning rate and scheduler durations. A larger effective batch is not guaranteed to improve final model quality.

### Q11. Does our current configuration give enough steps to both warm up and decay the learning rate?

**The current settings allocate almost the entire run to warmup.** This follows from counting optimizer updates separately from small-batch iterations.

The notebook currently sets:

```python
max_iters = 20000
gradient_accumulation_steps = 32
warmup_steps = 1000
```

`max_iters` counts small-batch iterations. The planned optimizer-update count is:

```python
optimizer_steps = (
    max_iters + gradient_accumulation_steps - 1
) // gradient_accumulation_steps
```

For these settings, that is exactly **20,000 / 32 = 625 updates**, assuming no mixed-precision overflow causes an optimizer update to be skipped. Since 20,000 is divisible by 32, every accumulation group is full.

The scheduler code then calculates:

```python
warmup_optimizer_steps = min(warmup_steps, max(1, optimizer_steps - 1))
```

Substituting the values:

```text
Warmup duration = min(1000, 624) = 624 scheduler steps
Decay duration  = max(1, 625 - 624) = 1 scheduler step
```

This provides **no sustained cosine-decay phase**. Because the scheduler advances after the weight update, the final call can set the minimum rate only after the last optimizer update; no subsequent training update uses that final rate in this run.

To obtain the intended warmup-then-decay behavior, warmup must occupy a substantially smaller part of the available optimizer-update budget, leaving many updates for decay. This is a scheduling issue in the current configuration, rather than a reason to remove gradient accumulation. The calculations here document the existing notebook settings; they do not change them.

### Q12. Why do we use gradient clipping, how does it work, and what happens without it?

**Gradient clipping limits unusually large gradients before an optimizer update.** It helps stabilize training when a batch produces a sudden gradient spike that could otherwise lead to an excessive change in the model parameters.

Our training loop uses:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

The default norm type is the L2 norm. Conceptually, take all parameter gradients together as one long vector `g`, then calculate its combined magnitude:

$$
\lVert g \rVert_2 = \sqrt{\sum_i g_i^2}
$$

The sum includes gradient elements across all parameters with gradients, rather than calculating a separate threshold for each layer. If the combined norm exceeds `0.5`, all gradients are scaled down by the same factor. Ignoring the implementation's small numerical-stability term, the operation is:

$$
g \leftarrow g \times \min\left(1, \frac{0.5}{\lVert g \rVert_2}\right)
$$

#### Example

```text
Combined gradient norm = 5.0
Maximum allowed norm   = 0.5
Scaling factor         = 0.5 / 5.0 = 0.1

Every gradient is multiplied by 0.1.
The resulting combined norm is approximately 0.5.
```

If the combined norm is already below the threshold, the gradients remain unchanged. Scaling all gradients by one positive factor preserves their overall direction while reducing their magnitude.

This operation **does not cap each individual gradient at 0.5**, and it does not clip the weights themselves. The trailing underscore in `clip_grad_norm_` indicates that it modifies the existing gradient tensors in place.

#### Where does clipping happen in our training loop?

After accumulating gradients across 32 microbatches, the update sequence is:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
scheduler.step()
```

The order matters:

1. **Accumulate gradients first:** the optimizer uses the combined gradient, so that is what we clip.
2. **Unscale before clipping:** when float16 loss scaling is enabled, restore the gradients to their actual scale so that `0.5` has the intended meaning. With the scaler disabled, this unscale operation is a no-op.
3. **Clip before the optimizer update:** the optimizer then receives the bounded gradients.
4. **Clear gradients after the update:** the next accumulation group starts without the previous group's gradients.

#### What happens without clipping?

Training can still work, especially if gradients remain well behaved. However, occasional large gradients can cause loss spikes or destabilize training. Clipping limits their influence; it does not fix every source of instability or guarantee better results.

A threshold that is too low can suppress useful gradients and slow learning. The value `0.5` is a training choice, rather than a universal requirement.

With AdamW, clipping bounds the **gradient norm**, rather than the exact parameter-update norm. AdamW also uses adaptive scaling, running gradient statistics, and weight decay, so the final weight change is not simply the clipped gradient multiplied by the learning rate.

### Q13. What does the AdamW optimizer do, and why do we use it?

**AdamW updates the model's trainable weights and biases using the gradients calculated by backpropagation.** Its updates aim to reduce training loss, although the loss need not decrease on every individual batch.

```text
Forward pass -> Calculate loss -> Backpropagate gradients
                                          |
                              Accumulate, unscale, and clip
                                          |
                                          v
                               AdamW updates parameters
```

Our configuration is:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-9,
)
```

AdamW combines a moving average of gradients, adaptive scaling based on squared gradients, and decoupled weight decay. These calculations operate element by element on the parameters.

#### 1. Remember recent gradient directions

For the gradient `g_t` at optimizer update `t`, AdamW maintains a first-moment estimate:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

This is an exponential moving average of gradients, often described as momentum. It smooths batch-to-batch noise while retaining recent directional information. Our `beta1 = 0.9`, so each update combines 90% of the previous average with 10% of the current gradient.

Here, `g_t` is the gradient supplied to AdamW after our accumulation and clipping steps, rather than necessarily the gradient from one microbatch.

#### 2. Track gradient scale and adapt the update

AdamW also maintains a second-moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

This tracks the moving average of squared gradients. It measures their recent scale; it is not a centered variance calculation. Our `beta2 = 0.95`, so each update combines 95% of the previous estimate with 5% of the current squared gradient.

Both estimates start at zero. AdamW corrects the resulting initial bias using:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$

The update direction is then scaled by `sqrt(v_hat) + epsilon`. This gives different parameter elements different effective update scales, while the scheduler still controls the shared learning rate for our optimizer group.

#### 3. Apply weight decay separately

As explained in Q8, weight decay shrinks parameters slightly. AdamW applies this separately from its gradient moment estimates, which is what "decoupled weight decay" means. This differs from feeding an L2 penalty gradient into standard Adam's moment calculations.

Combining decay and the adaptive gradient update gives:

$$
w_{t+1}
= (1-\eta_t\lambda)w_t
- \eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

| Symbol | Meaning in our setup |
| --- | --- |
| `w_t` | A model parameter before this optimizer update |
| `eta_t` | Current learning rate, controlled by the scheduler |
| `lambda` | Weight decay coefficient: `0.1` |
| `m_hat` | Bias-corrected moving average of gradients |
| `v_hat` | Bias-corrected moving average of squared gradients |
| `epsilon` | Numerical-stability constant: `1e-9` |

The first term applies decay; the second moves the parameter using the adaptive gradient direction. See the [PyTorch AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

#### Where does AdamW run in our loop?

```python
scaler.scale(loss).backward()  # Calculate and accumulate gradients

# After an accumulation group is complete:
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
scaler.step(optimizer)         # Invoke AdamW's parameter update
scaler.update()
optimizer.zero_grad(set_to_none=True)
scheduler.step()              # Set the learning rate for the next update
```

AdamW does not calculate gradients itself; `.backward()` does that. When enabled, `GradScaler` can skip the optimizer update if it detects nonfinite gradients. With the scaler disabled, `scaler.step(optimizer)` directly invokes the optimizer step.

#### Why choose AdamW, and what if we do not use it?

Momentum and adaptive scaling make AdamW useful for training Transformers, while its separate weight decay provides regularization. These are useful properties rather than a guarantee of successful training.

Without any optimizer update or equivalent manual parameter update, backpropagation would calculate gradients but the learned parameters would not change. Replacing AdamW with another optimizer, such as SGD, can still train the model, but its learning rate, momentum, schedule, and regularization may need different tuning.

AdamW also has a memory cost: it stores two running estimates per trainable parameter in addition to the parameters and gradients. Those estimates persist between optimizer updates, unlike activations computed for each new input.

### Q14. Which type of positional embedding do we use, and how does it work?

We use **learned absolute positional embeddings**. The model creates a trainable embedding table in `GPT.__init__()`:

```python
wpe = nn.Embedding(config.block_size, config.n_embd)
```

With our configuration, its shape is **[128, 384]**: one learned 384-dimensional vector for each position from **0 to 127**. This table contains **128 × 384 = 49,152 trainable parameters** and is separate from the token embedding table.

- **Learned:** the vectors are initialized when the model is created and updated through backpropagation during training.
- **Absolute:** each vector identifies a position within the current input window, such as position 0, 1, or 2, rather than directly representing the distance between two tokens.

In `forward()`, the model constructs position IDs and looks up their vectors:

```python
pos = torch.arange(0, t, dtype=torch.long, device=device)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

For a sequence of length `T`, `pos` contains `[0, 1, ..., T-1]`. The shapes are:

```text
Token embeddings:      [B, T, 384]
Position embeddings:      [T, 384]
                              |
                 Add, broadcasting across the batch
                              |
Combined embeddings:   [B, T, 384]
```

The token embedding identifies which token appears, while the position embedding supplies information about where it appears in the input. The same position table is used for every sequence in the batch. This is the GPT-2-style approach; our model does not implement fixed sinusoidal embeddings, relative positional embeddings, or rotary positional embeddings (RoPE).

#### What happens during generation beyond 128 tokens?

Our generation code crops the input to the latest 128 tokens before calling the model. `forward()` constructs positions starting at zero for each call, so those retained tokens receive positions **0 through 127 within the current window**, rather than their original indices in the full generated text.

The model can generate more than 128 tokens in total, but each forward pass uses at most 128 token representations and their corresponding learned absolute position vectors. The current position table has no entries for position IDs beyond 127.

### Q15. Are parameters updated once per batch in our training loop?

**Our parameters are normally updated once every 32 small batches, rather than after each one**, because we use gradient accumulation:

```python
batch_size = 32
gradient_accumulation_steps = 32
```

Each small batch, or microbatch, contains 32 sequences. Its forward pass calculates the loss, and its backward pass adds gradients to the existing gradient buffers. The weights remain unchanged until the accumulation group is complete:

```text
Microbatch 1  -> Calculate and accumulate gradients; weights unchanged
Microbatch 2  -> Calculate and accumulate gradients; weights unchanged
...
Microbatch 32 -> Calculate and accumulate gradients
              -> Unscale and clip gradients
              -> Update weights once
              -> Clear gradients for the next group
```

One full update therefore uses contributions from **32 × 32 = 1,024 sequences**, the effective batch size described in Q10. The optimizer updates parameters together using the accumulated gradients; it does not update them separately for each sequence or token.

Our loop decides when to update using:

```python
is_update_step = (
    (iteration + 1) % gradient_accumulation_steps == 0
    or iteration + 1 == max_iters
)
```

The parameter update is invoked inside that branch by:

```python
scaler.step(optimizer)
```

When float16 gradient scaling is enabled, the scaler can skip this update if nonfinite gradients are detected. Otherwise, it runs the AdamW update. With `gradient_accumulation_steps = 1`, the update branch would run after every small batch.

The final-iteration condition also processes an incomplete accumulation group if the run ends before collecting 32 microbatches. Our current `max_iters = 20000` is divisible by 32, so every group is full and there are **625 planned optimizer updates**. If the iteration count were changed to leave a partial group, the current fixed division of loss by 32 would scale that group's gradients below its own microbatch average.
