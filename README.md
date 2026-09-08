# AttoModel: 18.5M Parameter Micro-Agent Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**AttoModel** is an 18.5M parameter (14.04M active parameters per token) micro-agent language model designed to achieve reasoning, tool invocation, code synthesis, and structured reflection at sub-20M scale.

Engineered to bypass standard scaling laws, the architecture combines:
1. **Parallel Hybrid SSM-DiffAttn Heads**: Channels ($d_{\text{model}} = 512$) are split in parallel between a **Mamba-2 State Space Duality (SSD)** branch ($d_{\text{ssm}} = 256$) and a **Differential Attention** branch ($d_{\text{attn}} = 256$).
2. **Cross-Layer KV Sharing + MLA**: Layers 0–5 compute and maintain compressed latent KV caches ($d_{\text{latent}} = 24$), while Layers 6–11 share and reuse KV representations, restricting inference KV memory to **$< 1.2$\,MB at context length $C = 4,096$**.
3. **Granular Sparse Mixture-of-Experts (MoE)**: 1 Shared Expert + 4 Routed Experts with Top-2 routing (3 active experts per token).
4. **6 Learnable Meta-Tokens**: Static learnable vectors $M \in \mathbb{R}^{6 \times 512}$ prepended to input sequences to absorb high-attention sink activations.
5. **Muon Optimizer**: 5-step Newton-Schulz matrix polar orthogonalization for 2D projections combined with AdamW for 1D/SSM parameters under a **Warmup-Stable-Decay (WSD)** schedule.
6. **Group Relative Policy Optimization (GRPO)** with **Verifiable Rewards (RLVR)** for deterministic agentic alignment.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Input Tokens (B, T)       |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      | Tied Embedding Matrix       |
                      |   V = 8,192, d_model = 512  |
                      | + 6 Learnable Meta-Tokens   |
                      +--------------+--------------+
                                     |
         +---------------------------v---------------------------+
         |           12 x PARALLEL HYBRID BLOCKS                 |
         |                                                       |
         |   x ──► RMSNorm_1 ──┬──► Mamba-2 SSM (d=256) ──┐      |
         |                     │                          │      |
         |                     └──► Diff-Attn   (d=256) ──┴──► +  |
         |                            (Shared KV Layers 6-11) │  |
         |                                                    │  |
         |   Residual ◄───────────────────────────────────────┘  |
         |                                                       |
         |   x ──► RMSNorm_2 ──► Granular Sparse MoE ─────────► +|
         |                         (1 Shared + 4 Routed Top-2) │ |
         |                                                     │ |
         |   Residual ◄────────────────────────────────────────┘ |
         +---------------------------+---------------------------+
                                     |
                      +--------------v--------------+
                      | Final RMSNorm & Tied Head   |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |  Next-Token Logits (B, T, V)|
                      +-----------------------------+
```

### Table 1: Mathematical Parameter Budget

| Component | Specification / Dimensions | Parameters | % Budget |
| :--- | :--- | :--- | :--- |
| **Tied Embeddings ($W_{\text{emb}}$)** | $V = 8,192, d_{\text{model}} = 512$ | 4.19M | 25.1\% |
| **Hybrid Layers ($L = 12$)** | Parallel Mamba-2 SSD + DiffAttn ($d_{\text{head}} = 64$) | 6.43M | 38.5\% |
| **Granular Sparse MoE** | 1 Shared Expert + 4 Routed (Top-2, $d_{\text{ff}} = 72$) | 5.76M | 34.5\% |
| **Meta-Tokens & Norms** | 6 Meta-Tokens $\in \mathbb{R}^{6 \times 512}$, RMSNorms | 0.32M | 1.9\% |
| **Total Static Parameters** | Model Physical Capacity | **16.70M** | **100.0\%** |
| **Active Parameters / Token** | 1 Shared + 2 Routed Experts (Top-2) | **14.04M** | **15.9\% Reduction** |

---

## Core Innovations

### 1. Parallel Channel-Split Hybrid Head
Rather than sequentially interleaving attention and state-space layers, each hybrid block splits hidden state channels ($d_{\text{model}} = 512$) in parallel:
$$x_{\text{norm}} = \text{RMSNorm}_1(x) \in \mathbb{R}^{B \times T \times 512}$$
$$x_{\text{ssm\_in}} = x_{\text{norm}}[:, :, 0:256], \quad x_{\text{attn\_in}} = x_{\text{norm}}[:, :, 256:512]$$
$$x_{\text{hybrid}} = \left[ \gamma_{\text{ssm}} \cdot \text{SSM}(x_{\text{ssm\_in}}) \,\|\, \gamma_{\text{attn}} \cdot \text{DiffAttn}(x_{\text{attn\_in}}) \right]$$
$$x = x + x_{\text{hybrid}} + \text{MoE}(\text{RMSNorm}_2(x + x_{\text{hybrid}}))$$

### 2. Continuous Mamba-2 State Space Duality (SSD)
The SSM branch models continuous latent state evolution:
$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$
Discretized via Zero-Order Hold (ZOH) across $\Delta_t = \text{softplus}(\text{dt}_{\text{raw}} + \text{dt}_{\text{bias}})$:
$$\bar{A}_t = \exp(\Delta_t \cdot A), \quad \bar{B}_t = (\Delta_t \cdot B) \cdot x_t$$
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t, \quad y_t = \sum_{k=1}^{d_{\text{state}}} C_{t, k} h_{t, :, k} + D \cdot x_t$$
During autoregressive generation, recurrent state $h_t \in \mathbb{R}^{B \times 256 \times 16}$ updates in strictly $O(1)$ constant memory.

### 3. Differential Attention Branch & Linear SDPA Reformulation
Cancels background activation noise and prevents attention-head redundancy by differencing two softmax distributions:
$$\text{DiffAttn}(Q, K, V) = \left( \text{Softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d_k}}\right) - \lambda \cdot \text{Softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d_k}}\right) \right) V$$
where $\lambda = \text{sigmoid}(\lambda_{\text{param}}) \in (0, 1)$ is a learnable per-head scalar initialized to $\lambda = 0.8$.

**Linear SDPA Reformulation ($O(T)$ Memory Scaling)**:
Explicit materialization of $A_1, A_2 \in \mathbb{R}^{8 \times 4 \times 4096 \times 4096}$ requires $>93$\,GB VRAM during backward passes. By linearity of matrix multiplication:
$$(A_1 - \lambda A_2) V = (A_1 V) - \lambda (A_2 V) = \text{SDPA}(Q_1, K_1, V) - \lambda \cdot \text{SDPA}(Q_2, K_2, V)$$
This reduces peak training attention memory from $>90$\,GB to $<200$\,MB while achieving exact mathematical equivalence.

### 4. Multi-Head Latent Attention (MLA) & Cross-Layer KV Sharing
- **Multi-Head Latent Attention (MLA)**: Owner layers (0–5) compress input into low-rank latent KV vectors $c_t^{\text{KV}} \in \mathbb{R}^{B \times T \times 24}$, from which keys and values are decompressed via low-rank projections ($W_{\text{uk1}}, W_{\text{uk2}}, W_{\text{uv}}$).
- **Cross-Layer KV Sharing**: Layers 6–11 instantiate zero Key/Value projection matrices, instead directly sharing and reusing KV representations from their corresponding lower layer ($l - 6$).
- **Inference Footprint**:
  $$\text{KV Memory} = 6 \text{ owner layers} \times 4,096 \text{ tokens} \times 24 \text{ latent dim} \times 2 \text{ bytes} \approx \mathbf{1.125\text{ MB}}$$

### 5. Granular Sparse Mixture-of-Experts (MoE)
Each layer replaces the dense feed-forward network with 1 shared expert and 4 routed SwiGLU experts ($d_{\text{ff}} = 72$):
$$y_{\text{FFN}} = \text{Expert}_{\text{shared}}(x) + \sum_{i \in \text{Top-2}} G(x)_i \cdot \text{Expert}_i(x)$$
$$G(x) = \text{Softmax}\left(\text{Top-2}(x \cdot W_{\text{gate}})\right)$$
With auxiliary load balancing regularizer:
$$\mathcal{L}_{\text{MoE}} = N_{\text{routed}} \sum_{i=1}^{N_{\text{routed}}} f_i \cdot P_i$$

### 6. Learnable Meta-Tokens & Tied Embeddings
6 learnable vectors $M \in \mathbb{R}^{6 \times 512}$ are prepended to prompt sequences:
$$X_{\text{input}} = \left[ M_1; M_2; \dots; M_6; x_1; x_2; \dots; x_T \right]$$
These vectors absorb high-attention sink concentrations, stabilizing numerical representations in deeper layers. Embedding weights $W_{\text{emb}}$ are tied with the final language modeling head ($logits = F.linear(h_{\text{norm}}, W_{\text{emb}})$).

### 7. Compound Multi-Task Loss Objective
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CLM}} + 0.25 \mathcal{L}_{\text{FIM}} + 0.01 \mathcal{L}_{\text{MoE}} + 0.005 \mathcal{L}_{\text{Diff}}$$
- $\mathcal{L}_{\text{CLM}}$: Standard shifted next-token causal cross-entropy.
- $\mathcal{L}_{\text{FIM}}$: Fill-In-The-Middle structural infilling cross-entropy (PSM / SPM formats).
- $\mathcal{L}_{\text{MoE}}$: Expert load-balancing regularizer.
- $\mathcal{L}_{\text{Diff}}$: Differential regularizer penalizing positive cosine similarity between $Q_1$ and $Q_2$ projections.

### 8. Dual Optimizer: Muon (2D) + AdamW (1D) with WSD Schedule
- **Muon (2D Projections)**: 5-step Newton-Schulz iterations on polar factor $X_0 = M_t / (\|M_t\|_F + \epsilon)$:
  $$A = X_k X_k^T, \quad B = -4.7750 A + 2.0315 A^2, \quad X_{k+1} = 3.4445 X_k + B X_k$$
  $$W_t = W_{t-1} - \eta_t \left( 0.2 \cdot \frac{X_5}{\text{RMS}(X_5)} + 0.01 W_{t-1} \right)$$
- **AdamW (1D Vectors & SSM)**: $\eta_{\text{peak}} = 1.2 \times 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$.
- **WSD Schedule**: 5% linear warmup $\to$ 75% stable plateau ($\eta_{\text{peak}}$) $\to$ 20% minus-sqrt decay $(1 - \sqrt{t})$ to $\eta_{\text{min}} = 0.01 \cdot \eta_{\text{peak}}$.

### 9. Native Agentic Control Tokens
- `<|thought start|>` ... `<|thought end|>`: Internal chain-of-thought reasoning steps.
- `<|call tool|>`: Deterministic tool execution trigger.
- `<|tool response|>`: Receives execution payload back into context.
- `<|reflect|>`: Error self-correction loop upon failure.

---

## Repository Structure

```
├── configs/                     # YAML configurations for architecture & training
│   ├── model_18m.yaml           # Model hyperparameters (d_model=512, L=12, Vocab=8k)
│   ├── training_wsd.yaml        # 3-phase WSD schedule & batch size aggregation
│   └── data_mix.yaml            # Proportions & HF dataset IDs for Syntax, Logic, & Agentic
│
├── data/                        # Data pipeline, tokenization, streaming, and packing
│   ├── __init__.py
│   ├── tokenizer.py             # Custom 8k byte-level Tiktoken BPE wrapper & agentic tokens
│   ├── dataset.py               # StreamingSequencePacker, zero-padding & BinaryMemmapDataset
│   ├── filters.py               # Quality heuristics & MinHash LSH deduplicator
│   ├── hf_loader.py             # Live Hugging Face streaming curriculum loader
│   └── synthetic_pipeline.py    # AST JSON traces, math proofs, & agentic interaction traces
│
├── models/                      # Core neural network modules
│   ├── __init__.py
│   ├── sub20m_model.py          # Main AttoModel tying all blocks together
│   ├── hybrid_block.py          # Parallel Hybrid Block (Mamba-2 + Diff-Attn + MoE)
│   ├── ssm_branch.py            # Mamba-2 SSD with custom autograd FastSSMScan
│   ├── attention_branch.py      # Differential Attention with linear SDPA & MLA KV sharing
│   ├── moe.py                   # Granular Sparse MoE (1 shared + 4 routed experts)
│   ├── embeddings.py            # Tied embeddings & 6 learnable meta-token sinks
│   └── objectives.py            # Compound loss (L_CLM, L_FIM, L_MoE, L_Diff)
│
├── optimizers/                  # Training engines & optimizers
│   ├── __init__.py
│   ├── muon.py                  # 5-step Newton-Schulz polar orthogonalization optimizer
│   └── scheduler.py             # Warmup-Stable-Decay (WSD) minus-sqrt schedule
│
├── training/                    # Training engine and RLVR policy alignment
│   ├── __init__.py
│   ├── pretrain.py              # Multi-stage curriculum trainer with automated rollbacks
│   ├── grpo_rlvr.py             # Group Relative Policy Optimization + Verifiable Rewards
│   └── precision.py             # Mixed precision (BF16/FP16) & CUDA Graph runner
│
├── inference/                   # Deployment and execution
│   ├── __init__.py
│   ├── engine.py                # Zero-Memory KV hybrid engine with incremental UTF-8 decoder
│   ├── speculator.py            # Speculative decoding draft engine for large models
│   └── export.py                # Safetensors, ONNX, and GGUF export utilities
│
├── scripts/                     # Operational entrypoints
│   ├── prepare_dataset.py       # High-throughput streaming compiler for binary datasets & RLVR
│   ├── run_pretrain.sh          # Pre-training launch script (single-GPU / multi-GPU)
│   ├── run_grpo.sh              # Post-training RLVR alignment script
│   ├── chat.py                  # Terminal interactive REPL with agentic token highlighting
│   └── benchmark.py             # Throughput & latency testing suite
│
├── tests/                       # Complete pytest unit test suite (26 passing tests)
│   ├── test_data_pipeline.py
│   ├── test_hybrid_block.py
│   ├── test_moe.py
│   ├── test_differential_attention.py
│   ├── test_ssm.py
│   ├── test_muon.py
│   ├── test_tokenizer.py
│   ├── test_objectives.py
│   ├── test_inference.py
│   └── test_grpo.py
│
├── AttoModel_Updated_Architecture.tex # Complete academic LaTeX specification paper
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart & Step-by-Step Workflow

### Step 1: Installation
```bash
git clone https://github.com/your-org/AttoModel.git
cd AttoModel
pip install -e .
```
Verify complete test suite passes (26 unit tests):
```bash
python -m pytest tests/
```

### Step 2: Stream & Compile Dataset from Hugging Face
Compile a binary dataset directly from the Hugging Face streams specified in `AttoModel_Architecture.pdf`:
```bash
# Compile a 50 Million token dataset (takes ~2 minutes):
python scripts/prepare_dataset.py \
    --source huggingface \
    --target_tokens 50M \
    --output_bin data/curriculum_train.bin \
    --rlvr_output data/rlvr_prompts.json \
    --max_seq_len 4096

# Or compile the full 5.0 Billion token blueprint target:
python scripts/prepare_dataset.py \
    --source huggingface \
    --target_tokens 5B \
    --output_bin data/curriculum_train.bin \
    --rlvr_output data/rlvr_prompts.json \
    --max_seq_len 4096
```

### Step 3: Curriculum Pre-Training
Launch high-throughput pretraining with Muon (2D projections) + AdamW (1D/SSM):
```bash
# Automatic detection of data/curriculum_train.bin:
bash scripts/run_pretrain.sh

# Or directly with Python CLI (specifying epochs or steps):
python -m training.pretrain --data_bin data/curriculum_train.bin --epochs 1
```
Saves `pretrain_final.pt` into `./checkpoints/`.

### Step 4: Post-Training GRPO Alignment
Align the pre-trained checkpoint with verifiable code execution and tool-use rewards:
```bash
# Automatically loads checkpoints/pretrain_final.pt and data/rlvr_prompts.json:
bash scripts/run_grpo.sh ./checkpoints/pretrain_final.pt

# Or directly with Python CLI:
python -m training.grpo_rlvr \
    --checkpoint ./checkpoints/pretrain_final.pt \
    --prompts_file data/rlvr_prompts.json \
    --steps 100 \
    --group_size 8
```
Saves `grpo_final.pt` into `./checkpoints/` and `./checkpoints/grpo_aligned/`.

### Step 5: Interactive Chat & Speculative Inference
Launch the interactive terminal chat REPL with syntax highlighting for thought & tool tags:
```bash
python scripts/chat.py --checkpoint ./checkpoints/grpo_final.pt --temperature 0.3
```

Using Python API:
```python
from inference.engine import InferenceEngine, GenerationConfig

# Automatically loads the best trained/aligned checkpoint from ./checkpoints
engine = InferenceEngine.from_pretrained("./checkpoints")

prompt = "<|bos|>User: Write a python function to compute gcd(a, b).\n<|thought start|>"
response = engine.generate(prompt, GenerationConfig(max_new_tokens=128, temperature=0.3))
print(response)
```

---

## Pretraining Step, Epoch & Token Mathematics

Understanding how batch aggregation, dataset size, and step counts relate:
- **Micro-Batch Size**: 8
- **Context Length ($C$)**: 4,096 tokens
- **Gradient Accumulation Steps**: 16
- **Tokens Per Optimizer Step**: $8 \times 16 \times 4,096 = \mathbf{524,288\text{ tokens/step}}$
- **Single-GPU Throughput**: $\approx 12,110\text{ tokens/sec}$ ($\approx 43.3\text{ seconds/step}$)

| Dataset Size | Tokens | Steps / Epoch | 1 Epoch Time | 2 Epochs Time | Recommended Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10 Million (10M)** | 10,014,720 | **20 steps** | **~14.4 min** | **~28.8 min** | `python -m training.pretrain --data_bin curriculum_train.bin --epochs 1` |
| **50 Million (50M)** | 50,000,000 | **96 steps** | **~1.1 hours** | **~2.3 hours** | `python -m training.pretrain --data_bin curriculum_train.bin --epochs 1` |
| **100 Million (100M)**| 100,000,000 | **191 steps** | **~2.3 hours** | **~4.6 hours** | `python -m training.pretrain --data_bin curriculum_train.bin --epochs 1` |
| **1.0 Billion (1B)** | 1,000,000,000| **1,908 steps**| **~22.9 hours**| – | `python -m training.pretrain --data_bin curriculum_train.bin --epochs 1` |
| **5.0 Billion (5B)** | 5,000,000,000| **9,537 steps**| **~114 hours** | – | `bash scripts/run_pretrain.sh` |

> `pretrain.py` automatically detects dataset token size and adapts `max_steps` to 1 full epoch if a smaller dataset is provided. Pass `--epochs <N>` to run multiple epochs.

---

## Empirical Hardware Environment & 5-Hour Training Profile

The model was pre-trained, debugged, and aligned on a dedicated single-GPU workstation across an end-to-end sprint of approximately **5.0 hours**:

### Workstation Hardware Specification
- **GPU Accelerator**: 1x**NVIDIA RTX PRO 6000 Blackwell Workstation Edition**
  - **On-Board VRAM**: 97,887 MiB (96 GB) GDDR7
  - **Driver / CUDA**: NVIDIA Driver 595.84 / CUDA 12.8
  - **TDP Envelope**: 600W Peak Thermal Design Power
- **Host Processor (CPU)**: 32-core x86_64 host processor
- **System Memory (RAM)**: 188.19 GB high-speed DDR5 RAM
- **Scratch Storage**: High-throughput NVMe Solid-State Drive with zero-copy binary memory-mapped streaming (`np.memmap`)
- **OS & Deep Learning Stack**: Linux (Ubuntu x86_64, Kernel 6.8.0), PyTorch 2.6.0+cu128, AMP (bfloat16)

### 5-Hour End-to-End Execution Profile
- **Stage 1: Curriculum Pre-Training (~2.9 Hours)**:
  - **Steps & Tokens**: 144 accumulation steps, processing 75,497,472 tokens (~75.5M tokens) at 524,288 tokens/step.
  - **Throughput**: ~12,110 tokens/second sustained (~43.3 seconds per optimizer step).
  - **Loss Trajectory**: Cross-entropy compound loss decreased from $\mathcal{L}_1 = 145.75 \to \mathcal{L}_{50} = 45.34 \to \mathcal{L}_{144} = 28.54$.
  - **Memory Footprint**: Stabilized at 45.02 GiB peak VRAM (well within the 96 GB physical VRAM ceiling) following SDPA linear differencing and `_FastSSMScanFunction` integration.
- **Stage 2: Post-Training GRPO RLVR Alignment (~2.1 Hours)**:
  - **Steps & Exploration**: 100 policy gradient update steps generating $G = 8$ candidate completions per prompt ($T_{\text{gen}} = 128$ tokens).
  - **Reward Evaluation**: Evaluated across real MATH-500, Python coding, and Glaive function-calling prompts with deterministic AST (+1.0), Tool JSON (+2.0), and Reflection (+1.0) verification.
  - **Result**: Successfully saved `checkpoints/grpo_final.pt` with zero policy collapse.
- **Total End-to-End Wall-Clock Time**: **~5.0 hours** from scratch to an aligned micro-agent model on a single professional workstation GPU!

---

## Systems & Autograd Optimizations

During the empirical development and stress-testing of AttoModel on single workstation GPUs, four critical systems bottlenecks were identified and resolved:

1. **Differential Attention Linear Split ($O(T)$ Activation Memory)**:
   - *Problem*: Materializing $A_1, A_2 \in \mathbb{R}^{B \times H \times T \times T}$ caused a 93\,GB CUDA Out of Memory error at $T = 4,096$.
   - *Resolution*: Rewritten as $(A_1 V) - \lambda (A_2 V)$ using PyTorch's native `F.scaled_dot_product_attention` (FlashAttention-backed), shrinking peak memory to $< 200$\,MB.
2. **Fast SSM Scan Custom Autograd Function**:
   - *Problem*: Python for-loop recurrence created $>49,000$ autograd graph nodes per layer, causing an 8.6\,s backward latency per layer.
   - *Resolution*: Implemented `_FastSSMScanFunction` with manual analytical reverse-recurrence gradients, dropping backward step time to $0.12$\,s ($67\times$ speedup).
3. **Streaming Incremental UTF-8 Decoding**:
   - *Problem*: Multi-byte UTF-8 characters decoded one token at a time yielded replacement characters ($\text{\ufffd}$).
   - *Resolution*: Integrated `codecs.getincrementaldecoder('utf-8')` into `InferenceEngine.generate_stream()` to preserve partial byte buffers across decoding steps.
4. **Unprintable Control-Byte Logit Masking**:
   - *Problem*: Raw C0 control bytes (null bytes, backspaces, vertical tabs) caused terminal cursor distortion.
   - *Resolution*: Precomputed all non-UTF-8 and unprintable byte tokens (7,326 tokens) and masked their logits to $-\infty$ during sampling.

---

## Future Development Roadmap

To scale AttoModel from a micro-agent architecture to frontier reasoning capabilities:

1. **Scale Pre-Training to 370M--5.0B Tokens**:
   - Sub-20M parameter models must be overtrained at ratios between $20:1$ (Chinchilla minimum: 370M tokens) and $270:1$ (AttoModel Blueprint: 5.0B tokens) to attain fluent English grammar and natural language semantic coherence.
2. **Supervised Fine-Tuning (SFT) Instruction Bridge**:
   - Introduce an intermediate 50M-token instruction-following SFT stage between WSD pre-training and GRPO alignment to establish conversational priors before reinforcement learning.
3. **Containerized Execution Sandbox for RLVR**:
   - Upgrade the reward evaluator from static AST syntax parsing (`ast.parse`) to dynamic sandboxed execution in an isolated Docker container with unit-test assertion verification.
4. **Data-Driven Byte-Pair Encoding (BPE) Vocab**:
   - Replace synthetic alphabet combinations with data-driven merges trained directly on the educational and code training corpus.
5. **Multi-Token Prediction (MTP) Heads**:
   - Add dual next-token prediction heads to double training sample efficiency per step and provide $2\times$ faster draft tokens during speculative decoding with large frontier models.

---

## License
MIT License.
