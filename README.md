# Compiled Context Inference

> Pre-compiled KV states for zero-prefill inference, thought injection,
> and autonomous self-improvement.

## Status: What's Proven, What's Not

**Proven and working:**
- Pre-compiled KV states eliminate prefill latency (content processed once, queried instantly)
- RoPE de-rotation enables order-independent composition of compiled states (42/42 tests)
- Thought injection via compiled KV states steers model reasoning (HumanEval 79.9% → 97.0%)
- Self-steering loop with strategy library (90.9% autonomous on HumanEval)
- Qwen3.6-35B-A3B MoE at 33 tok/s with state save/restore via llama-cpp-python
- Retrieval + compilation pipeline: 100K corpus, 5ms search, 32 tok/s generation

**Not proven:**
- "Zero context window" — the model's active knowledge IS bounded by GPU VRAM.
  Compiled KV states are pre-computed but still occupy VRAM during generation.
  This system manages the context window more efficiently; it does not eliminate it.
- FFN-level knowledge injection (compiling knowledge into model weights without
  training) — mechanism exists, partially tested, does not yet deliver content
  comprehension. This is active research, not a proven capability.
- The earlier 1.58M-token benchmark was retrieval + compilation (CPU search found
  the needle, relevant chunk compiled into KV), not the model holding 1.58M tokens
  in state. The model knew ~200 tokens. The system accessed 1.58M.

## What This Does

Compile content (code, documents, conversation) into persistent KV states
stored in system RAM. Save to disk. Reload instantly. The model attends to
pre-compiled content without re-processing it. Combined with CPU search over
indexed corpora, this provides fast access to large knowledge bases with
constant generation speed.

### Proven Results

| What | Result | Hardware |
|---|---|---|
| Compiled KV state recall | 4/4 facts across 4 queries from saved state | RTX 3080, 5x M40 |
| Prefill elimination | 0ms prefill from saved state (vs minutes for 1M+ tokens) | RTX 3080 |
| RoPE de-rotation | Order-independent composition, 42/42 tests | RTX 3080 |
| 35B MoE generation | 33.3 tok/s with state save/restore | 5x M40 24GB |
| Retrieval pipeline | 100K corpus, 5ms search, 32 tok/s | 5x M40 24GB |
| HumanEval (thought injection) | 97.0% (7B, zero training) | RTX 3080 |
| HumanEval (self-steering) | 90.9% autonomous | RTX 3080 |

### Limitations

- Active compiled state is bounded by GPU VRAM (~56 KB/token for 7B)
- For content beyond VRAM: requires retrieval step (CPU search → compile relevant chunk)
- The retrieval approach is functionally RAG with pre-compiled KV states instead of prompt stuffing
- FFN-level knowledge injection (true "zero context window") is unproven for comprehension

## Quick Start

```bash
pip install -r requirements.txt

# Interactive chat with compiled context
python compiled_chat.py --model Qwen/Qwen2.5-7B-Instruct

# Commands inside chat:
#   /compile_file src/main.py
#   /compiled
#   "Explain the main function"
```

### OpenAI-Compatible API Server

```bash
python compiled_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

curl -X POST http://localhost:8000/v1/compile \
  -d '{"id": "facts", "text": "The planet Zargthorp has 3 moons."}'

curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What are the moons of Zargthorp?"}]}'
```

### Programmatic Usage

```python
from compiled_engine import CompiledInference

engine = CompiledInference("Qwen/Qwen2.5-7B-Instruct")
engine.compile("codebase", open("src/main.py").read())
response = engine.chat("What does the main function do?")
engine.save("./compiled_states/")
engine.load("./compiled_states/")
```

## Architecture

Three components:

1. **Compiler** (`ContextCompiler`) — forward pass on content, saves KV states to CPU RAM.
   RoPE de-rotation stores position-neutral keys for order-independent composition.

2. **Composer** (`ContextComposer`) — selects compiled states, assigns sequential positions,
   re-rotates keys, loads to GPU. Compose any subset in any order.

3. **Generator** (`ContextGenerator`) — generates from composed state. Zero prefill latency.

### Control Mechanisms (All Proven)

| # | Mechanism | What it does |
|---|---|---|
| 1 | Compiled KV states | Pre-compute content, reload instantly |
| 2 | System-role compilation | Compile behavioral directives into KV states |
| 3 | Multi-layer steering | Override trained facts via W_lm injection at 9 layers |
| 4 | Logit bias | Verbatim reproduction on quantized models |
| 5 | Thought injection | Inject reasoning hints mid-generation via compiled KV |

### Self-Steering Loop

Autonomous retry: generate → test → on failure, select strategy from ranked
library → inject as thought → retry. 18 strategies, success rates tracked
across sessions.

## Files

| File | Purpose |
|------|---------|
| `compiled_context.py` | Core: Compiler, Composer, Generator, RoPE de-rotation |
| `compiled_engine.py` | Production engine with tools, streaming, persistence |
| `compiled_server.py` | OpenAI-compatible API server |
| `compiled_chat.py` | Interactive CLI |
| `self_steering.py` | Autonomous strategy selection and retry |
| `test_rope_derotation.py` | RoPE de-rotation test suite (42 tests) |
| `test_35b_llamacpp.py` | 35B MoE compiled context via llama.cpp (33 tok/s) |
| `test_100k_pathb.py` | 100K corpus retrieval + compilation pipeline |
| `bench_humaneval*.py` | HumanEval benchmark scripts |
| `docs/ARCHITECTURE.md` | Technical architecture |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Any HuggingFace causal language model
- For hybrid attention models (Qwen3.5/3.6): `llama-cpp-python` + GGUF
- GPU with VRAM for model weights + compiled states
- System RAM for state storage (~56 KB per token for 7B)
