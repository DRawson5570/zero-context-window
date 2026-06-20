"""conversation_channel.py — Compiled conversation with iterative multi-turn support.

Each turn's user message and model response are compiled into steerer
features and a pre-built attention mask.  The model sees a short header
rather than the full conversation text.  The KV cache stays constant.
The steerer injects conversation structure.

Usage:
    from conversation_channel import CompiledConversation
    conv = CompiledConversation(tokenizer)

    # Turn 1
    ctx = conv.add_turn("Explain how attention works.", "Attention is...")
    steerer.set_weights(ctx['features'])
    # model input: ctx['header'] = "[CTX:compiled:1 turn, 45 tokens]"
    # model attention mask: ctx['attention']

    # Turn 2
    ctx = conv.add_turn("Explain the KVQ mechanism in detail.", response)
    # header now: "[CTX:compiled:2 turns, 120 tokens]"

Author: DeepSeek v4 Pro · 2026-06-19
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# CompiledConversation
# ---------------------------------------------------------------------------

_CONV_WINDOW = 32  # local attention window within each message


class CompiledConversation:
    """Multi-turn conversation with compiled features and attention mask.

    Each turn's user message and assistant response accumulate into a
    running compiled state.  The model receives a short header token
    while all structural context (roles, turn boundaries, Q/A edges,
    cross-turn entity references) is injected via steerer features
    and pre-computed attention edges.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._turns: list[dict] = []  # {user, assistant, user_tokens, asst_tokens}
        self._total_tokens = 0
        self._feature_dim = 4  # [is_user, turn_num, is_question, qa_distance]

    def _tokenize(self, text: str) -> list[int]:
        enc = self._tok(text, return_offsets_mapping=True,
                        add_special_tokens=False)
        return enc.input_ids, enc.offset_mapping

    def add_turn(self, user_message: str,
                 assistant_response: str = "") -> dict:
        """Add a new turn to the conversation.

        Args:
            user_message: The user's question/input.
            assistant_response: The model's previous response (empty on
                first turn before generation).

        Returns:
            {
                'header': str,     # "[CTX:compiled:N turns, M tokens]"
                'features': torch.Tensor,  # [total_tokens, 4] per-token features
                'attention': torch.Tensor, # [total_tokens, total_tokens] mask
                'turn_count': int,
            }
        """
        user_ids, _ = self._tokenize(user_message)
        asst_ids, _ = self._tokenize(assistant_response) if assistant_response else ([], [])

        self._turns.append({
            'user': user_message,
            'assistant': assistant_response,
            'user_len': len(user_ids),
            'asst_len': len(asst_ids),
        })
        self._total_tokens += len(user_ids) + len(asst_ids)

        return self._compile()

    def _compile(self) -> dict:
        """Build the compiled conversation state."""
        n = len(self._turns)
        T = self._total_tokens

        features = torch.zeros(T, self._feature_dim, dtype=torch.float32)
        attention = torch.zeros(T, T, dtype=torch.float32)

        offset = 0
        for turn_idx, turn in enumerate(self._turns):
            u_len = turn['user_len']
            a_len = turn['asst_len']
            turn_norm = turn_idx / max(n, 1)

            # User tokens: features
            if u_len > 0:
                features[offset:offset + u_len, 0] = 1.0       # is_user
                features[offset:offset + u_len, 1] = turn_norm # turn number
                features[offset:offset + u_len, 2] = 1.0       # is_question
                # Q/A distance: counts up from 0 for each token in the question
                for i in range(u_len):
                    features[offset + i, 3] = i / max(u_len, 1)

            # Assistant tokens: features
            a_start = offset + u_len
            if a_len > 0:
                features[a_start:a_start + a_len, 0] = 0.0      # not user
                features[a_start:a_start + a_len, 1] = turn_norm
                features[a_start:a_start + a_len, 2] = 0.0      # not question
                for i in range(a_len):
                    features[a_start + i, 3] = (u_len + i) / max(u_len + a_len, 1)

            # Attention: local window within each message
            for msg_start, msg_len in [(offset, u_len), (a_start, a_len)]:
                for i_token in range(msg_len):
                    pos = msg_start + i_token
                    lo = max(msg_start, pos - _CONV_WINDOW + 1)
                    attention[pos, lo:pos + 1] = 1.0

            # Attention: Q → A edges (question tokens attend to answer tokens)
            if u_len > 0 and a_len > 0:
                # Each position in the answer attends to the matching position in the question
                for i_a in range(min(a_len, u_len)):
                    attention[a_start + i_a, offset + i_a] = 1.0
                # Answer tail attends to question tail
                mid_a = a_start + a_len // 2
                attention[mid_a:, offset:offset + u_len] = 1.0

            # Attention: cross-turn links (current user start → previous assistant end)
            if turn_idx > 0:
                prev = self._turns[turn_idx - 1]
                prev_asst_end = offset
                prev_asst_start = max(0, prev_asst_end - prev['asst_len'])
                if prev_asst_end > prev_asst_start and u_len > 0:
                    attention[offset:offset + min(4, u_len),
                              prev_asst_start:prev_asst_end] = 1.0

            offset += u_len + a_len

        header = f"[CTX:compiled:{n} turns, {T} tokens]"

        return {
            'header': header,
            'features': features,
            'attention': attention,
            'turn_count': n,
        }

    def reset(self):
        """Clear conversation history."""
        self._turns.clear()
        self._total_tokens = 0

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# ConversationRetriever — entity-aware context scanner
# ---------------------------------------------------------------------------

class ConversationRetriever:
    """Scans conversation history for entity mentions and injects context.

    When a new user question arrives, key terms are extracted and matched
    against the full conversation text.  Matching contexts are injected
    into the prompt so the model can recall specific facts from any depth
    without expanding the attention window.

    Zero training.  Zero model changes.  Pure string matching — fast and
    deterministic.  Fits extracted context within the prompt's attention
    window.

    Usage:
        >>> ret = ConversationRetriever(tokenizer)
        >>> ret.add_turn("Explain attention. Paris is the capital of France.")
        >>> ret.add_turn("How does attention relate to transformers?")
        >>> ctx = ret.retrieve_and_inject("What did we say about Paris?")
        >>> print(ctx[:200])
    """

    CONTEXT_TOKENS = 80   # tokens of surrounding context per match

    def __init__(self, tokenizer, max_matches: int = 3):
        self._tok = tokenizer
        self._conversation_text: list[str] = []  # raw text per turn
        self._max_matches = max_matches

    def add_turn(self, user_message: str, assistant_response: str = ""):
        """Store a turn's raw text for future retrieval."""
        turn_text = f"User: {user_message}"
        if assistant_response:
            turn_text += f"\nAssistant: {assistant_response}"
        self._conversation_text.append(turn_text)

    def _extract_terms(self, question: str) -> list[str]:
        """Extract key terms from a question for matching.

        Keeps capitalized words (entities), nouns > 3 chars, and
        significant noun phrases.
        """
        words = question.split()
        terms = []
        for w in words:
            clean = w.strip('.,?!;:"\')(').lower()
            if len(clean) > 3:
                terms.append(clean)
        # Add hyphenated and capitalized terms
        caps = [w.strip('.,?!') for w in words if w[0].isupper() and len(w) > 3]
        terms.extend(caps)
        return list(dict.fromkeys(terms))  # deduplicate, preserve order

    def _find_context(self, term: str) -> str:
        """Find the first mention of a term in conversation history and
        return surrounding context tokens."""
        full_text = '\n'.join(self._conversation_text)
        pos = full_text.lower().find(term.lower())
        if pos < 0:
            return ""

        # Take CONTEXT_TOKENS tokens around the match position
        before = full_text[max(0, pos - 200):pos]
        after = full_text[pos:pos + 300]
        context = before.split()[-self.CONTEXT_TOKENS:] + ['[...]'] + after.split()[:self.CONTEXT_TOKENS]
        return ' '.join(context)

    def retrieve_and_inject(self, user_question: str) -> str:
        """Build a prompt with retrieved context injected.

        Extracts terms from the question, finds them in conversation
        history, injects matching contexts into the prompt so the model
        sees specific facts about the user's query.
        """
        terms = self._extract_terms(user_question)
        contexts = []
        seen = set()

        for term in terms:
            if len(contexts) >= self._max_matches:
                break
            ctx = self._find_context(term)
            if ctx and term.lower() not in seen:
                contexts.append(ctx)
                seen.add(term.lower())

        if not contexts:
            return user_question

        # Build enriched prompt
        lines = ['[Context retrieved from conversation history:]']
        for i, ctx in enumerate(contexts):
            lines.append(f'  Earlier mention: ...{ctx}...')
        lines.append(f'---')
        lines.append(f'Current question: {user_question}')
        return '\n'.join(lines)

    def reset(self):
        self._conversation_text.clear()


# ---------------------------------------------------------------------------
# FactInjector — compile conversation facts into FFN-layer embeddings
# ---------------------------------------------------------------------------

class FactInjector:
    """Compile conversation facts into steering vectors from lm_head weights.

    Extracts normalized lm_head.weight rows for each fact token, then
    injects them as steering vectors at the fact-band layer for a few
    decode steps.  The model's hidden state is pushed toward the correct
    token distribution — zero training, pre-compiled from the model's
    own weights.

    Usage:
        >>> fi = FactInjector(model, tokenizer, layer=14)
        >>> fi.add_fact("Paris is the capital of France.")
        >>> # Register hooks before generation, remove after
        >>> fi.register()
        >>> response = model.generate(...)
        >>> fi.remove()
    """

    def __init__(self, model, tokenizer, layer: int = 14, max_steps: int = 8, alpha: float = 300.0):
        self._model = model
        self._tok = tokenizer
        self._layer = layer
        self._max_steps = max_steps
        self._alpha = alpha

        # Extract lm_head weight for steering vectors
        self._W_lm = self._get_lm_head().weight.data.float()

        self._fact_vectors: dict[str, list[torch.Tensor]] = {}
        self._orig_attn = None
        self._wrapper = None
        self._steps = 0

    def _get_lm_head(self):
        for attr in ['lm_head', 'model.lm_head', 'model.decoder.lm_head']:
            parts = attr.split('.')
            obj = self._model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise AttributeError("Cannot find lm_head in model")

    def _get_layer(self):
        for attr in ['model.layers', 'layers', 'model.decoder.layers',
                     'transformer.h', 'h']:
            parts = attr.split('.')
            obj = self._model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        return []

    def add_fact(self, fact_text: str):
        """Tokenize a fact and store normalized lm_head vectors for each token."""
        ids = self._tok.encode(fact_text, add_special_tokens=False)
        if not ids:
            return
        vectors = []
        for tid in ids:
            v = self._W_lm[tid].clone().detach()
            nrm = v.norm()
            vectors.append(v / nrm if nrm > 0 else v)
        # Extract query terms from first few words for matching
        words = fact_text.split()
        for w in words[:4]:
            key = w.strip('.,?!;:').lower()
            if len(key) > 2:
                self._fact_vectors[key] = vectors

    def add_facts(self, facts: list[str]):
        for f in facts:
            self.add_fact(f)

    def add_conversation(self, turns: list[dict]):
        """Compile entire conversation into a dense steering vector."""
        all_vectors = []
        for turn in turns:
            for role in ('user', 'assistant'):
                text = turn.get(role, '')
                ids = self._tok.encode(text, add_special_tokens=False)
                for tid in ids:
                    v = self._W_lm[tid].clone().detach()
                    nrm = v.norm()
                    if nrm > 0:
                        all_vectors.append(v / nrm)
        if all_vectors:
            stacked = torch.stack(all_vectors)
            self._all_vectors = stacked
            self._dense_vector = stacked.mean(dim=0)
            self._dense_vector = self._dense_vector / self._dense_vector.norm().clamp(min=1e-8)
            return True
        return False

    def inject_for_question(self, question: str, max_vectors: int = 12):
        """Select conversation vectors most relevant to the question and inject sequentially.

        Computes cosine similarity between the question's lm_head vector
        and every conversation token's vector.  Selects the top-N most
        similar for sequential injection (matching the demo's pattern).
        The model is steered toward the correct context one step at a time.
        """
        if not hasattr(self, '_all_vectors'):
            return False
        # Encode question as vector for similarity comparison
        q_ids = self._tok.encode(question, add_special_tokens=False)
        if not q_ids:
            return False
        q_vecs = []
        for tid in q_ids[:20]:  # limit to first 20 tokens of question
            v = self._W_lm[tid].clone().detach()
            nrm = v.norm()
            if nrm > 0:
                q_vecs.append(v / nrm)
        if not q_vecs:
            return False
        q_avg = torch.stack(q_vecs).mean(dim=0)

        # Cosine similarity with all conversation vectors
        all_v = self._all_vectors.to(device=q_avg.device)
        sims = torch.mv(all_v, q_avg)  # dot product (vectors already unit-norm)
        top_idx = sims.topk(min(max_vectors, len(sims))).indices
        top_idx = top_idx[top_idx.argsort()]  # preserve chronological order

        selected = [self._all_vectors[i] for i in top_idx]
        self._wrapper._vectors = selected
        self._wrapper._step = 0
        self._wrapper._single_vector_mode = False
        return True
        if hasattr(self, '_dense_vector'):
            self._wrapper._vectors = [self._dense_vector]
            self._wrapper._step = 0
            self._wrapper._single_vector_mode = True
            return True
        return False

    def register(self):
        """Replace the fact-band attention layer with a steering wrapper."""
        if self._wrapper is not None:
            return
        layers = self._get_layer()
        if not layers or self._layer >= len(layers):
            return
        self._orig_attn = layers[self._layer].self_attn
        self._wrapper = _SteeringWrapper(self._orig_attn, self)
        layers[self._layer].self_attn = self._wrapper
        self._step = 0

    def remove(self):
        if self._orig_attn is not None and self._wrapper is not None:
            layers = self._get_layer()
            if layers and self._layer < len(layers):
                layers[self._layer].self_attn = self._orig_attn
        self._orig_attn = None
        self._wrapper = None

    def inject_facts_from_question(self, question: str):
        """Extract terms from the question, match stored facts, set injection."""
        terms = [w.strip('.,?!;:').lower() for w in question.split() if len(w) > 3]
        all_vectors = []
        for term in terms:
            if term in self._fact_vectors:
                all_vectors.extend(self._fact_vectors[term])
        if all_vectors:
            self._wrapper._vectors = all_vectors[:10]  # cap at 10 vectors
            self._wrapper._step = 0  # reset step counter
        return len(all_vectors) > 0

    def reset_facts(self):
        self._fact_vectors.clear()
        self.remove()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


class _SteeringWrapper(torch.nn.Module):
    """Wraps an attention layer, injecting steering vectors at output.

    Matches the compiled cartridge injection pattern from the demo:
    - Post-processes the OUTPUT of original attention
    - Injects only at last position [-1] (autoregressive decode)
    - One steering vector per decode step, sequential with decay
    """

    def __init__(self, orig_attn, injector: FactInjector):
        super().__init__()
        self._orig = orig_attn
        self._injector = injector
        self._vectors: list[torch.Tensor] = []
        self._step = 0
        self._single_vector_mode = False

    def forward(self, hidden_states, *args, **kwargs):
        out = self._orig(hidden_states, *args, **kwargs)
        tensor_out = out[0] if isinstance(out, tuple) else out
        injector = self._injector

        if self._single_vector_mode and self._vectors:
            # Dense mode: same vector every step, no decay
            v = self._vectors[0].to(device=tensor_out.device, dtype=torch.float32)
            tensor_out = tensor_out.clone()
            tensor_out[..., -1, :] += injector._alpha * v
            return (tensor_out,) + out[1:] if isinstance(out, tuple) else tensor_out

        if self._step < min(len(self._vectors), injector._max_steps):
            v = self._vectors[self._step].to(
                device=tensor_out.device, dtype=torch.float32)
            decay = max(0.0, 1.0 - 0.12 * self._step)
            boost = 1.5 if self._step == 0 else 1.0
            tensor_out = tensor_out.clone()
            tensor_out[..., -1, :] += injector._alpha * decay * boost * v
            self._step += 1
            return (tensor_out,) + out[1:] if isinstance(out, tuple) else tensor_out
        return out
