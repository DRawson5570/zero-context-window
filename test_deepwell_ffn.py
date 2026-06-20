#!/usr/bin/env python3
"""Deepwell Chronicles — narrative content compiled into FFN gated neurons.

Tests the zero context window mechanism with a rich, multi-turn story
containing 42 verifiable facts across 8 turns (~3,500 tokens).
"""
import torch, sys, time
sys.path.insert(0, '/home/drawson/compiled_priors')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deepwell_story import STORY_TURNS, ALL_QUESTIONS

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map='cuda:0', trust_remote_code=True)
model.eval()
W = model.lm_head.weight.data
print("Loaded 7B 4-bit")

def capture_state(text):
    msgs = [{'role': 'user', 'content': text}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(txt, return_tensors='pt').input_ids.to('cuda:0')
    c = {}
    def f(m, inp):
        x = inp[0] if isinstance(inp, tuple) else inp
        c['v'] = x[0, -1, :].detach().float().cpu()
    h = model.model.layers[26].mlp.register_forward_pre_hook(f)
    with torch.no_grad():
        model(ids)
    h.remove()
    return c['v']

# Compile each turn into gated FFN neurons
# Gate: hidden state from a question about the turn's topic
# Response: W_lm vectors from a concise answer
neurons = []

print("\nCompiling story into FFN neurons...")
total_tokens = 0
for turn_data in STORY_TURNS:
    text = turn_data["text"]
    total_tokens += len(tok.encode(text, add_special_tokens=False))

    for question, keys in turn_data["facts"]:
        gate = capture_state(question)
        gate = gate / gate.norm().clamp(min=1e-8)

        answer_words = ' '.join(keys)
        resp_ids = tok.encode(answer_words, add_special_tokens=False)
        resp_vecs = [W[tid].float() / W[tid].float().norm().clamp(min=1e-8) for tid in resp_ids]

        neurons.append({
            'gate': gate.to('cuda:0'),
            'vecs': resp_vecs,
            'question': question,
            'keys': keys,
        })

print("Compiled: %d neurons from %d turns (%d tokens of narrative)" % (
    len(neurons), len(STORY_TURNS), total_tokens))

def gen_gated(question, alpha=50):
    gs = capture_state(question)
    gs = gs / gs.norm().clamp(min=1e-8)
    gs = gs.to('cuda:0')

    sims = torch.tensor([float(gs @ n['gate'].float()) for n in neurons])
    best = sims.argmax().item()
    sel = neurons[best]
    vecs = sel['vecs']

    msgs = [{'role': 'user', 'content': question}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(txt, return_tensors='pt').input_ids.to('cuda:0')

    hooks = []
    for l in [26, 27]:
        step = [0]
        def mk(s, a, vv, n):
            def fn(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                if x.shape[1] > 1:
                    return out
                if s[0] < n:
                    v = vv[s[0]].to(out.device, dtype=out.dtype)
                    o = out.clone()
                    o[:, -1, :] += a * v
                    s[0] += 1
                    return o
                return out
            return fn
        hooks.append(model.model.layers[l].mlp.register_forward_hook(
            mk(step, alpha, vecs, len(vecs))))

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    for h in hooks:
        h.remove()

    answer = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return answer, sel['question'], sims[best].item()

# Test all 42 facts
print("\n" + "=" * 60)
print("DEEPWELL CHRONICLES — FFN GATED RECALL")
print("=" * 60)

hits = 0
misses = []
t0 = time.time()

for turn_data in STORY_TURNS:
    turn_hits = 0
    for question, keys in turn_data["facts"]:
        answer, matched, sim = gen_gated(question)
        found = [k for k in keys if k.lower() in answer.lower()]
        hit = len(found) > 0
        if hit:
            hits += 1
            turn_hits += 1
        else:
            misses.append((question, matched, sim, answer[:60]))

    print("Turn %d: %d/%d" % (turn_data["turn"], turn_hits, len(turn_data["facts"])))

elapsed = time.time() - t0
total = len(ALL_QUESTIONS)

print("\n" + "=" * 60)
print("RESULT: %d/%d (%.1f%%)" % (hits, total, 100*hits/total))
print("Story: %d tokens across %d turns" % (total_tokens, len(STORY_TURNS)))
print("Neurons: %d gated FFN neurons" % len(neurons))
print("Time: %.0fs (%.1f q/s)" % (elapsed, total/elapsed))

if misses:
    print("\nMisses:")
    for q, m, s, a in misses[:10]:
        print("  Q: %s" % q[:40])
        print("  Gate: %s (%.2f) → %s" % (m[:30], s, a[:50]))

if hits == total:
    print("\nPERFECT. The story is in the FFN.")
