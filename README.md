# State-Modular Neural Runtime

The model knows what it needs to, when it needs to.

## The Architecture: Two Paths, One System

### Path A: Compiled KV States (Comprehension)

The model READS content by processing it through its forward pass. The
resulting KV states are saved, reloaded instantly, and the model attends to
them — genuine comprehension. This is bounded by the context window (n_ctx)
but gives full reasoning capability.

The model asks to read content via triggers:
```
Model: "Let me check that file. [READ:src/auth.py]"
System: reads file → compiles into KV cache via forward pass
Model: "The authenticate function validates JWTs and has a timing issue..."
```

Proven: 26.7 tok/s at 40K context on Qwen3.6-35B-A3B (5x M40 24GB).
State save/restore in 0.3s. Disk persistence. RoPE de-rotation for
order-independent composition (42/42 tests).

### Path B: Steerer + Cartridges (Awareness AND Reasoning)

The model REASONS about content without attending to it. Compiled features
(AST structure, topic vectors, scope depths) are injected directly into the
residual stream via the steerer at early/mid layers. The subsequent FFN
layers — which act as key-value memory banks — READ these features, trigger
associations, and produce genuine reasoning. The late attention layers
handle local syntax. The raw content is never in the KV cache.

This is not pattern matching. The injected features flow through 50+
layers of FFN computation. The model's own knowledge retrieval mechanism
operates on the semantic coordinates of the content. It reasons in latent
space.

For 599 lines of code the model never attended to:
1. `CodeChannelComputer` parses the AST → 5-channel structural features
2. Steerer injects features into residual stream at layers 10-20
3. FFN layers 21-60 read these features → reason about the code
4. Attention handles local token relationships only
5. The model generates analysis. The 599 lines were never in the KV cache.

- O(1) per token — no KV cache growth
- Constant 21 MB active cache regardless of corpus size
- Attention layers never see the compiled content
- **The model reasons through FFN, not attention**
- Scales to unlimited content

Proven: 820M steered backbone at PPL 36.9 (below oracle ceiling).
Cartridge injection verbatim at alpha=400. O(T×W) sliding window +
steerer matches full O(T²) attention.

### The Bridge: Triggers

Path B tells the model WHAT EXISTS. Path A lets the model READ IT.
Triggers connect them:

```
Path B (steerer features) → model knows "this project has auth.py,
                             database.py, 50 API endpoints"
User asks about authentication
Model emits → [READ:src/auth.py]
Path A (compile into KV) → file processed through forward pass
Model reasons → "The authenticate function validates JWTs..."
```

The model navigates with awareness (B) and comprehends with attention (A).
Just like you know your codebase's structure without reading every file.
When you need details, you open the specific file.

### Available Triggers

| Trigger | Action |
|---|---|
| `[READ:path]` | Read file, compile into KV cache |
| `[SEARCH:query]` | Search accumulated context |
| `[FILE:path]` | Structural summary (AST) |
| `[SYMBOL:name]` | Find definition across repo |
| `[CALC:expr]` | Math evaluation |
| `[QUERY:entity]` | Compiled PPMI lookup |

## Proven Results

| What | Result |
|---|---|
| 35B MoE generation | 26.7 tok/s at 40K context, 33.3 tok/s at 8K |
| State save/restore | 0.3s from RAM, 0.31s from disk |
| RoPE de-rotation | Order-independent composition, 42/42 tests |
| Trigger-aware reading | Model emits [READ:], reads files, analyzes correctly |
| Retrieval pipeline | 100K corpus, 5ms search, 32 tok/s |
| HumanEval (thought injection) | 97.0% (7B, zero training) |
| HumanEval (self-steering) | 90.9% autonomous |
| Steered backbone | PPL 36.9, below oracle ceiling |
| Cartridge injection | Verbatim fact reproduction at alpha=400 |
| O(T×W) attention | Matches full O(T²) with steerer carrying global context |

## What This Is and What It Isn't

**What it is:** A runtime with two reasoning pathways. Path B injects
compiled features into the residual stream where the FFN layers reason
about them — genuine computation on content the model never attended to.
Path A compiles content into KV states for precise, detail-level
comprehension. Triggers let the model navigate from awareness (B) to
reading (A) on demand. The system has no context window.

**What it isn't:** Magic. Path B reasoning operates on compiled structural
features, not raw text. The quality of reasoning depends on the quality
of the compiled features. Path A comprehension is bounded by the context
window. Together they cover the full spectrum — but each path has its
own strengths and limits.

## Quick Start

```bash
pip install -r requirements.txt

python compiled_chat.py --model Qwen/Qwen2.5-7B-Instruct

# Commands:
#   /compile_file src/main.py
#   /compiled
#   "Explain the main function"
```

### API Server (35B on pe2)

```bash
python server_35b.py --port 8000

# The model reads files on demand:
curl http://pe2:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Read /tmp/sample.py and explain what it does."}]}'

# Pre-compile content:
curl -X POST http://pe2:8000/v1/compile \
  -d '{"id":"docs","text":"Your content here"}'
```

## Architecture Components

| Component | Path | What it does |
|---|---|---|
| `ContextCompiler` | A | Forward pass → KV states with RoPE de-rotation |
| `ContextComposer` | A | Order-independent state composition |
| `ContextGenerator` | A | Generate from composed states |
| `SuperpositionSteererV3` | B | 22K params, injects features at 9 layers |
| `FactInjector` | B | W_lm cartridge at fact-band layer |
| `_CartridgeProcessor` | B | Logit bias for verbatim output |
| `CodeChannelComputer` | B | AST features from source code |
| `ConversationRetriever` | A+B | Text search across history |
| Trigger system | Bridge | Model emits [READ:], [SEARCH:], etc. |
| `SelfSteeringLoop` | A | Autonomous retry with strategy library |

## Files

| File | Purpose |
|------|---------|
| `compiled_context.py` | Path A: Compiler, Composer, Generator, RoPE de-rotation |
| `compiled_engine.py` | Path A: Production engine with tools, streaming |
| `compiled_server.py` | Path A: OpenAI-compatible API (7B HuggingFace) |
| `server_35b.py` | Path A+Bridge: Trigger-aware server (35B llama.cpp) |
| `compiled_chat.py` | Interactive CLI |
| `conversation_channel.py` | Path B: FactInjector, ConversationRetriever |
| `code_channel.py` | Path B: AST features for code awareness |
| `code_tools.py` | Bridge: Model-callable triggers |
| `self_steering.py` | Autonomous strategy selection and retry |
| `test_rope_derotation.py` | RoPE de-rotation test suite (42 tests) |
| `test_35b_llamacpp.py` | 35B compiled context proof |
| `test_100k_pathb.py` | 100K retrieval pipeline |
| `bench_humaneval*.py` | HumanEval benchmarks |
| `docs/ARCHITECTURE.md` | Technical architecture |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for HuggingFace backend / Path A+B on 7B)
- For 35B hybrid models: `llama-cpp-python` + GGUF (Path A + Bridge)
- GPU with VRAM for model weights + active compiled states
- System RAM for content storage and compiled state caching
