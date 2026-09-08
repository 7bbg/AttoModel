```markdown
micro_agent_model/
│
├── configs/                     # YAML/JSON configurations for models & training
│   ├── model_18m.yaml           # Hyperparameters (d_model=512, L=12, Vocab=8k)
│   ├── training_wsd.yaml        # 3-phase WSD schedule & batch size ramps
│   └── data_mix.yaml            # Proportions for FineWeb-Edu, Synthetic, & CoT
│
├── data/                        # Data pipeline, tokenization, and packing
│   ├── tokenizer.py             # Custom 8k byte-level Tiktoken BPE wrapper
│   ├── dataset.py               # Zero-padding sequence packing & block masking
│   ├── filters.py               # FastText classifier & MinHash LSH deduplication
│   └── synthetic_pipeline.py    # AST / JSON trace formatting utilities
│
├── models/                      # Core neural network modules
│   ├── __init__.py
│   ├── sub20m_model.py          # Main model class tying all blocks together
│   ├── hybrid_block.py          # Parallel Hybrid-Head (Mamba-2 + Diff-Attn)
│   ├── ssm_branch.py            # Mamba-2 chunked scan wrapper (O(1) state)
│   ├── attention_branch.py      # Differential Attention & MLA compressive head
│   ├── moe.py                   # Granular Sparse MoE (1 shared + 4 routed experts)
│   ├── embeddings.py            # Tied input/output embeddings & learnable meta-tokens
│   └── objectives.py            # Compound loss ($L_{CLM}$, $L_{FIM}$, $L_{MoE}$, $L_{Diff}$)
│
├── optimizers/                  # Training engines & optimizers
│   ├── muon.py                  # Newton-Schulz orthogonalized 2D matrix updates
│   └── scheduler.py             # Warmup-Stable-Decay (WSD) & minus-square-root annealing
│
├── training/                    # Training loop and RL alignment
│   ├── pretrain.py              # Multi-stage curriculum trainer with automated rollbacks
│   ├── grpo_rlvr.py             # Group Relative Policy Optimization + Verifiable Rewards
│   └── precision.py             # FP8 mixed precision & checkpointing utilities
│
├── inference/                   # Deployment and execution
│   ├── engine.py                # High-speed inference & KV cache management (Zero-Memory KV)
│   ├── speculator.py            # Speculative decoding draft engine for large models
│   └── export.py                # GGUF / ExecuTorch conversion scripts
│
├── scripts/                     # Operational entrypoints
│   ├── run_pretrain.sh          # Multi-GPU cluster training launch script
│   ├── run_grpo.sh              # RLVR alignment script
│   └── benchmark.py             # Throughput & latency testing suite
│
├── tests/                       # Unit tests for components (MoE routing, Diff-Attn, Muon)
│   ├── test_hybrid_block.py
│   ├── test_moe.py
│   └── test_muon.py
│
├── README.md
├── pyproject.toml
└── requirements.txt

```