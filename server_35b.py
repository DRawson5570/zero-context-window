#!/usr/bin/env python3
"""State-Modular Neural Runtime — OpenAI-compatible server.

The model knows what it needs to, when it needs to.
Content lives as text in RAM. When the model needs details, it emits
a trigger ([READ:path], [SEARCH:query]). The system compiles the
content into the model's KV cache on the fly. The model continues
with genuine knowledge.

Launch:
    export LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
    python3 server_35b.py [--port 8000] [--host 0.0.0.0]
"""
import os, sys, time, json, uuid, re, argparse, glob as globmod
from typing import Optional

os.environ["LLAMA_CPP_LIB_PATH"] = "/usr/local/lib/ollama"
LIB_DIR = "/usr/local/lib/ollama"
ld = os.environ.get("LD_LIBRARY_PATH", "")
if LIB_DIR not in ld:
    os.environ["LD_LIBRARY_PATH"] = LIB_DIR + ":" + LIB_DIR + "/cuda_v12:" + ld

import ctypes
_ggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)
_ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
_ggml.ggml_backend_load_all_from_path.restype = None
_ggml.ggml_backend_load_all_from_path(LIB_DIR.encode("utf-8"))
_ggml.ggml_backend_load_all_from_path((LIB_DIR + "/cuda_v12").encode("utf-8"))

from llama_cpp import Llama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

GGUF = "/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"

SYSTEM_PROMPT = """You are a helpful assistant with access to a content library.

You can read files and search your memory using these commands:
- [READ:path] — Read a file from disk. The file content will be loaded into your memory.
- [SEARCH:query] — Search your accumulated context for relevant information.

When you need to reference a file or recall prior context, emit the appropriate command.
The system will load the content and you can then reason about it naturally.

Example: "Let me check that file. [READ:src/auth.py]"
The file will appear in your context and you can analyze it."""

app = FastAPI(title="State-Modular Neural Runtime")
llm: Optional[Llama] = None
compiled_states: dict = {}
context_store: list = []


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3.6:35b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    presence_penalty: float = 0.0

class CompileRequest(BaseModel):
    id: str
    text: str


def build_prompt(messages, system=SYSTEM_PROMPT):
    parts = []
    if system:
        parts.append("<|im_start|>system\n%s<|im_end|>" % system)
    for msg in messages:
        parts.append("<|im_start|>%s\n%s<|im_end|>" % (msg.role, msg.content))
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "\n".join(parts)


def detect_trigger(text):
    m = re.search(r'\[([A-Z]+):([^\]]+)\]', text)
    if m:
        return m.group(1), m.group(2), m.start(), m.end()
    return None


def handle_trigger(kind, payload):
    if kind == "READ":
        path = payload.strip()
        if not os.path.exists(path):
            return "[File not found: %s]" % path
        try:
            with open(path, 'r') as f:
                content = f.read()
            if len(content) > 20000:
                content = content[:20000] + "\n... [truncated at 20K chars]"
            context_store.append({"type": "file", "path": path, "content": content})
            return "\n--- Content of %s ---\n%s\n--- End of %s ---\n" % (path, content, path)
        except Exception as e:
            return "[Error reading %s: %s]" % (path, e)

    elif kind == "SEARCH":
        query = payload.strip().lower()
        results = []
        for item in context_store:
            text = item.get("content", "")
            if query in text.lower():
                snippet = text[:500]
                results.append(snippet)
        for cid, state in compiled_states.items():
            if query in cid.lower():
                results.append("[Compiled state: %s]" % cid)
        if not results:
            return "[No results for: %s]" % payload
        return "\n--- Search results for '%s' ---\n%s\n--- End results ---\n" % (
            payload, "\n...\n".join(results[:3]))

    return "[Unknown trigger: %s]" % kind


def generate_with_triggers(prompt, request):
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.reset()

    for sid, state in compiled_states.items():
        llm.load_state(state)
        break

    llm.eval(tokens)

    generated = []
    pending_text = ""
    t0 = time.time()

    for _ in range(request.max_tokens):
        token = llm.sample(
            temp=request.temperature,
            top_p=request.top_p,
            repeat_penalty=1.0 + request.presence_penalty,
        )
        if token == llm.token_eos():
            break

        generated.append(token)
        chunk = llm.detokenize([token]).decode("utf-8", errors="replace")
        pending_text += chunk
        llm.eval([token])

        trigger = detect_trigger(pending_text)
        if trigger:
            kind, payload, start, end = trigger
            result = handle_trigger(kind, payload)
            result_tokens = llm.tokenize(result.encode("utf-8"), add_bos=False)
            llm.eval(result_tokens)
            result_token_ids = result_tokens
            generated.extend(result_token_ids)
            pending_text = ""

    elapsed = time.time() - t0
    text = llm.detokenize(generated).decode("utf-8", errors="replace")
    text = text.replace("<|im_end|>", "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    gen_count = len(generated)
    return text, gen_count, gen_count / elapsed if elapsed > 0 else 0


async def stream_with_triggers(prompt, request):
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.reset()

    for sid, state in compiled_states.items():
        llm.load_state(state)
        break

    llm.eval(tokens)
    chat_id = "chatcmpl-" + uuid.uuid4().hex[:8]
    in_think = True
    pending_text = ""

    for _ in range(request.max_tokens):
        token = llm.sample(
            temp=request.temperature,
            top_p=request.top_p,
            repeat_penalty=1.0 + request.presence_penalty,
        )
        if token == llm.token_eos():
            break

        text = llm.detokenize([token]).decode("utf-8", errors="replace")
        llm.eval([token])
        pending_text += text

        trigger = detect_trigger(pending_text)
        if trigger:
            kind, payload, start, end = trigger
            result = handle_trigger(kind, payload)
            result_tokens = llm.tokenize(result.encode("utf-8"), add_bos=False)
            llm.eval(result_tokens)
            pending_text = ""
            continue

        if in_think:
            if "</think>" in pending_text:
                in_think = False
                pending_text = ""
            continue

        if text.replace("<|im_end|>", ""):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": text.replace("<|im_end|>", "")}, "finish_reason": None}],
            }
            yield "data: %s\n\n" % json.dumps(chunk)

    yield "data: %s\n\n" % json.dumps({
        "id": chat_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"


@app.on_event("startup")
async def startup():
    global llm
    print("Loading model...")
    t0 = time.time()
    llm = Llama(
        model_path=GGUF,
        n_gpu_layers=-1,
        n_ctx=8192,
        n_threads=20,
        n_threads_batch=40,
        verbose=False,
        use_mmap=True,
    )
    print("Model loaded in %.1fs" % (time.time() - t0))


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen3.6:35b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    prompt = build_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            stream_with_triggers(prompt, request),
            media_type="text/event-stream",
        )

    text, n_tokens, tps = generate_with_triggers(prompt, request)

    context_store.append({
        "type": "conversation",
        "user": request.messages[-1].content if request.messages else "",
        "assistant": text[:500],
        "content": request.messages[-1].content + " " + text[:500],
    })

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": n_tokens, "total_tokens": n_tokens},
    }


@app.post("/v1/compile")
async def compile_content(request: CompileRequest):
    prompt = "<|im_start|>system\n%s<|im_end|>\n" % request.text
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)

    llm.reset()
    t0 = time.time()
    llm.eval(tokens)
    compile_time = time.time() - t0

    state = llm.save_state()
    compiled_states[request.id] = state
    state_size = len(state.llama_state)

    context_store.append({"type": "compiled", "id": request.id, "content": request.text})

    return {
        "id": request.id,
        "tokens": len(tokens),
        "compile_time_ms": int(compile_time * 1000),
        "state_size_mb": round(state_size / 1e6, 1),
    }


@app.get("/v1/compiled")
async def list_compiled():
    items = {}
    for cid, state in compiled_states.items():
        items[cid] = {"n_tokens": state.n_tokens, "state_size_mb": round(len(state.llama_state) / 1e6, 1)}
    return {"compiled_states": items, "count": len(items)}


@app.delete("/v1/compiled/{content_id}")
async def delete_compiled(content_id: str):
    if content_id in compiled_states:
        del compiled_states[content_id]
        return {"deleted": content_id}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/v1/context")
async def list_context():
    return {
        "items": len(context_store),
        "types": {t: sum(1 for c in context_store if c.get("type") == t)
                  for t in set(c.get("type", "") for c in context_store)},
    }


@app.get("/v1/engine/stats")
async def engine_stats():
    return {
        "model": "Qwen3.6-35B-A3B",
        "backend": "llama-cpp-python + Ollama libllama.so + cuda_v12",
        "compiled_states": len(compiled_states),
        "context_items": len(context_store),
        "n_ctx": 8192,
        "triggers": ["READ", "SEARCH"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
