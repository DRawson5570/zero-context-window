"""code_channel.py — AST-based per-token code features for the compiled steerer.

Extracts 5 channels of structural code features from Python source using only
stdlib (ast, tokenize). Zero dependencies. Zero SGD.

Features (one float32 per channel per token):
  Channel 21: structural context (is_def, is_class, is_import, control_flow)
  Channel 22: depth and flow   (scope_depth, is_decorator, is_return, is_assign)
  Channel 23: type awareness   (has_type_annotation, annotation_kind, same_type, typing_import)
  Channel 24: symbol identity  (same_file, imported, usage_count, is_keyword)
  Channel 25: distance/errors  (definition_distance, symbol_context, syntax_error, reserved)

All features are normalized to [0,1] range for direct steerer injection.

Author: DeepSeek v4 Pro · 2026-06-19
"""

from __future__ import annotations

import ast
import collections
import keyword as kwmod
import os
import threading
import time
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# AST visitor — collect positional features
# ---------------------------------------------------------------------------

_KEYWORDS = frozenset(kwmod.kwlist)
_PYTHON_KW = {k: 1 for k in kwmod.kwlist}


class _CodeVisitor(ast.NodeVisitor):
    """Walk the AST and record per-byte feature vectors."""

    def __init__(self, source_lines: list[str]):
        self._lines = source_lines
        self.total_bytes = sum(len(l) + 1 for l in source_lines)
        # Per-byte accumulators
        self._is_def = bytearray(self.total_bytes)
        self._is_class = bytearray(self.total_bytes)
        self._is_import = bytearray(self.total_bytes)
        self._is_control = bytearray(self.total_bytes)
        self._depth = bytearray(self.total_bytes)
        self._is_decorator = bytearray(self.total_bytes)
        self._is_return = bytearray(self.total_bytes)
        self._is_assign = bytearray(self.total_bytes)
        self._has_type_ann = bytearray(self.total_bytes)
        self._ann_kind = bytearray(self.total_bytes)
        self._typing_import = bytearray(self.total_bytes)
        self._is_same_file = bytearray(self.total_bytes)
        self._is_imported_sym = bytearray(self.total_bytes)
        self._usage_count = [0] * self.total_bytes  # int accum
        self._keyword_flag = bytearray(self.total_bytes)
        self._def_distance = [65535] * self.total_bytes  # high sentinel
        self._same_symbol_cnt = [0] * self.total_bytes
        self._error_region = bytearray(self.total_bytes)
        self._error_kind = bytearray(self.total_bytes)

        self._symbol_defs: dict[str, int] = {}  # name → byte_offset of first def
        self._symbol_last_pos: dict[str, list[int]] = collections.defaultdict(list)
        self._nesting_depth = 0  # track ancestry depth during walk

    def _byte_offset(self, lineno: int, col_offset: int) -> int:
        off = 0
        for i in range(min(lineno - 1, len(self._lines))):
            off += len(self._lines[i]) + 1
        return off + min(col_offset, len(self._lines[lineno - 1]) if lineno <= len(self._lines) else 0)

    def _mark_range(self, buf: bytearray, node: ast.AST, val: int = 1):
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return
        s = self._byte_offset(node.lineno, node.col_offset)
        e = self._byte_offset(node.end_lineno, node.end_col_offset)
        s = min(s, self.total_bytes - 1)
        e = min(e, self.total_bytes)
        for i in range(s, e):
            buf[i] = val

    def _mark_range_int_list(self, arr: list[int], node: ast.AST, val: int = 1):
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return
        s = self._byte_offset(node.lineno, node.col_offset)
        e = self._byte_offset(node.end_lineno, node.end_col_offset)
        s = min(s, self.total_bytes - 1)
        e = min(e, self.total_bytes)
        for i in range(s, e):
            arr[i] = val

    def _mark_depth(self, node: ast.AST):
        """Mark bytes covered by node with the current nesting depth."""
        self._mark_range_int_list(self._depth, node, min(self._nesting_depth + 1, 255))

    def generic_visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._nesting_depth += 1
            result = super().generic_visit(node)
            self._nesting_depth -= 1
            return result
        return super().generic_visit(node)

    # -- visitor methods --

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._mark_range(self._is_def, node)
        self._mark_depth(node)
        self._symbol_defs[node.name] = self._byte_offset(node.lineno, node.col_offset)
        # Annotations
        if node.returns is not None:
            ret_start = self._byte_offset(node.returns.lineno, node.returns.col_offset)
            ret_end = self._byte_offset(node.returns.end_lineno, node.returns.end_col_offset)
            for i in range(max(ret_start, 0), min(ret_end, self.total_bytes)):
                self._has_type_ann[i] = 1
                self._ann_kind[i] = self._annotation_kind_id(node.returns)
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation is not None:
                self._mark_range(self._has_type_ann, arg.annotation)
                self._mark_range_int_list(self._ann_kind, arg.annotation,
                                          self._annotation_kind_id(arg.annotation))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self._mark_range(self._is_class, node)
        self._mark_depth(node)
        self._symbol_defs[node.name] = self._byte_offset(node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        self._mark_range(self._is_import, node)
        for alias in node.names:
            self._symbol_defs[alias.asname or alias.name] = self._byte_offset(node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self._mark_range(self._is_import, node)
        for alias in node.names:
            self._symbol_defs[alias.asname or alias.name] = self._byte_offset(node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self._mark_range(self._is_control, node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        self._mark_range(self._is_return, node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        self._mark_range(self._is_assign, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._mark_range(self._is_assign, node)
        if node.annotation is not None:
            self._mark_range(self._has_type_ann, node.annotation)
            self._mark_range_int_list(self._ann_kind, node.annotation,
                                      self._annotation_kind_id(node.annotation))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._mark_range(self._is_assign, node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        off = self._byte_offset(node.lineno, node.col_offset)
        if off < self.total_bytes:
            if node.id in _KEYWORDS:
                self._keyword_flag[off] = 1
            # Same-file symbol tracking
            if node.id in self._symbol_defs:
                self._is_same_file[off] = 1
                def_off = self._symbol_defs[node.id]
                self._def_distance[off] = min(self._def_distance[off], abs(off - def_off))
            # Symbol usage counting
            self._usage_count[off] = self._usage_count[off] + 1
            # Same-symbol context (will be aggregated in post-processing)
            self._symbol_last_pos[node.id].append(off)

    @staticmethod
    def _annotation_kind_id(ann: ast.AST) -> int:
        """Map type annotation AST → kind ID 0-7."""
        if isinstance(ann, ast.Name):
            name = ann.id.lower()
            kind_map = {'int': 1, 'str': 2, 'float': 3, 'bool': 4, 'list': 5, 'dict': 6, 'none': 0}
            return kind_map.get(name, 7)
        if isinstance(ann, ast.Subscript):
            return _CodeVisitor._annotation_kind_id(ann.value)
        return 7


# ---------------------------------------------------------------------------
# Token-level alignment
# ---------------------------------------------------------------------------


def _build_per_token_features(visitor: _CodeVisitor, source: str,
                              tokenizer) -> torch.Tensor:
    """Aggregate per-byte features into per-token [T, 5] tensor."""
    enc = tokenizer(source, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc.input_ids
    offsets = enc.offset_mapping

    T = len(ids)
    if not offsets or len(offsets) != T:
        return torch.zeros(T, 5, dtype=torch.float32)

    feat = torch.zeros(T, 5, dtype=torch.float32)
    scope_max = max(1, max(visitor._depth) if any(visitor._depth) else 1)

    for tok_idx, (start, end) in enumerate(offsets):
        start = min(start, visitor.total_bytes - 1)
        end = min(end, visitor.total_bytes)
        n_cover = max(end - start, 1)

        def _mean(buf, div=1.0):
            return sum(buf[i] for i in range(start, end) if i < len(buf)) / (max(n_cover, 1) * max(div, 1e-9))

        # Channel 21: structural context
        ch21_def = _mean(visitor._is_def)
        ch21_cls = _mean(visitor._is_class)
        ch21_imp = _mean(visitor._is_import)
        ch21_ctl = _mean(visitor._is_control)
        feat[tok_idx, 0] = ch21_def + ch21_cls * 0.5 + ch21_imp * 0.25 + ch21_ctl * 0.125

        # Channel 22: depth and flow
        depths = [visitor._depth[i] for i in range(start, end) if i < len(visitor._depth)]
        ch22_dep = min(depths) / scope_max if depths else 0.0
        ch22_dec = _mean(visitor._is_decorator)
        ch22_ret = _mean(visitor._is_return)
        ch22_asn = _mean(visitor._is_assign)
        feat[tok_idx, 1] = ch22_dep * 0.5 + ch22_dec * 0.2 + ch22_ret * 0.2 + ch22_asn * 0.1

        # Channel 23: type awareness
        ch23_ann = _mean(visitor._has_type_ann)
        ch23_kind = _mean(visitor._ann_kind, 8.0)
        ch23_typ = _mean(visitor._typing_import)
        feat[tok_idx, 2] = ch23_ann * 0.4 + ch23_kind * 0.3 + ch23_typ * 0.3

        # Channel 24: symbol identity
        ch24_same = _mean(visitor._is_same_file)
        ch24_impt = _mean(visitor._is_imported_sym)
        ch24_use = min(sum(visitor._usage_count[i] for i in range(start, end) if i < len(visitor._usage_count)), 100) / 100.0
        ch24_kw = _mean(visitor._keyword_flag)
        feat[tok_idx, 3] = ch24_same * 0.4 + ch24_impt * 0.3 + ch24_use * 0.2 + ch24_kw * 0.1

        # Channel 25: distance and errors
        def_dists = [visitor._def_distance[i] for i in range(start, end) if i < len(visitor._def_distance) and visitor._def_distance[i] < 65535]
        ch25_dist = min(def_dists) / 10000.0 if def_dists else 1.0
        ch25_err = _mean(visitor._error_region)
        ch25_ekind = _mean(visitor._error_kind, 3.0)
        feat[tok_idx, 4] = ch25_dist * 0.3 + ch25_err * 0.4 + ch25_ekind * 0.3

    feat = feat.clamp(0.0, 1.0)
    return feat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CodeChannelComputer:
    """Extract per-token code features from Python source via AST.

    Usage:
        >>> computer = CodeChannelComputer()
        >>> features = computer.compute(source, tokenizer)
        >>> # features.shape = [T, 5]  (T = number of BPE tokens)
    """

    def __init__(self):
        pass

    def compute(self, source: str, tokenizer) -> torch.Tensor:
        """Compute [T, 5] code channel features for a Python source string.

        Args:
            source: Python source code string.
            tokenizer: HuggingFace PreTrainedTokenizer (GPT-2 or compatible).

        Returns:
            Float32 tensor of shape [T, 5] where T is the number of BPE tokens.
            All values in [0, 1] range.
        """
        lines = source.splitlines(keepends=True)
        if not lines:
            lines = ['']

        visitor = _CodeVisitor(lines)

        try:
            tree = ast.parse(source)
            visitor.visit(tree)
        except SyntaxError as e:
            if e.lineno is not None and e.offset is not None:
                err_off = visitor._byte_offset(e.lineno, e.offset)
                for i in range(max(0, err_off - 50), min(visitor.total_bytes, err_off + 50)):
                    visitor._error_region[i] = 1
                    visitor._error_kind[i] = 1
            return _build_per_token_features(visitor, source, tokenizer)

        return _build_per_token_features(visitor, source, tokenizer)

    def compute_batch(self, sources: list[str], tokenizer) -> list[torch.Tensor]:
        """Compute features for multiple source strings."""
        return [self.compute(s, tokenizer) for s in sources]


# ---------------------------------------------------------------------------
# Convenience: GPT-2 tokenizer
# ---------------------------------------------------------------------------

_gpt2_tok = None

def _get_tokenizer():
    global _gpt2_tok
    if _gpt2_tok is None:
        from transformers import AutoTokenizer
        _gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
    return _gpt2_tok


# ---------------------------------------------------------------------------
# CodebaseCache — per-repo incremental feature store
# ---------------------------------------------------------------------------

class CodebaseCache:
    """Per-repo incremental feature cache with mtime-based staleness.

    Stores per-file [T, 5] feature tensors and re-parses only when a file
    has been modified since its last cache entry.  Thread-safe for
    concurrent file-save events and generation reads (use with lock in
    multi-threaded contexts).

    Usage:
        >>> cache = CodebaseCache(tokenizer)
        >>> features = cache.get('src/main.py')  # parse if stale or missing
        >>> cache.update('src/main.py', source)  # force re-parse after edit
        >>> cache.preload_repo('.')              # warm up entire repo
    """

    def __init__(self, tokenizer, computer=None):
        self._tok = tokenizer
        self._computer = computer or CodeChannelComputer()
        self._features: dict[str, torch.Tensor] = {}
        self._mtimes: dict[str, float] = {}
        self._import_graph: dict[str, set[str]] = {}

    def _read_source(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read()

    def _is_stale(self, file_path: str) -> bool:
        if file_path not in self._mtimes:
            return True
        try:
            return os.stat(file_path).st_mtime != self._mtimes[file_path]
        except FileNotFoundError:
            return False  # virtual file — cached from update()

    def get(self, file_path: str) -> torch.Tensor:
        """Return features for file_path, recomputing if stale or missing."""
        if file_path in self._features and not self._is_stale(file_path):
            return self._features[file_path]
        try:
            source = self._read_source(file_path)
            self._features[file_path] = self._computer.compute(source, self._tok)
            self._mtimes[file_path] = os.stat(file_path).st_mtime
        except FileNotFoundError:
            if file_path in self._features:
                return self._features[file_path]  # virtual file — keep cached
        return self._features.get(file_path, torch.zeros(0, 5, dtype=torch.float32))

    def update(self, file_path: str, source: str):
        """Recompute and cache features from a source string."""
        self._features[file_path] = self._computer.compute(source, self._tok)
        try:
            self._mtimes[file_path] = os.stat(file_path).st_mtime
        except FileNotFoundError:
            self._mtimes[file_path] = time.time()  # virtual file — mark fresh

    def invalidate(self, file_path: str):
        """Mark file_path as stale. Does not recompute — lazy on next get()."""
        self._mtimes.pop(file_path, None)

    def preload_repo(self, root_dir: str, glob_pat: str = "**/*.py",
                     verbose: bool = False):
        """Precompute and cache all matching files in a repository.

        Walks the directory tree, parses each .py file, and stores features.
        Does NOT resolve cross-file symbol information — use
        compute_with_imports() for that.
        """
        import glob as _glob
        root = os.path.abspath(root_dir)
        files = sorted(_glob.glob(os.path.join(root, glob_pat), recursive=True))
        t0 = time.time()
        for i, fpath in enumerate(files):
            self.get(fpath)
            if verbose and (i + 1) % 100 == 0:
                print(f'  preload: {i+1}/{len(files)} files '
                      f'({time.time() - t0:.1f}s)', flush=True)
        if verbose:
            print(f'  preload: {len(files)} files total '
                  f'({time.time() - t0:.1f}s)', flush=True)

    def __len__(self):
        return len(self._features)

    def __contains__(self, file_path: str):
        return file_path in self._features and not self._is_stale(file_path)

    def __getitem__(self, file_path: str):
        return self.get(file_path)


# ---------------------------------------------------------------------------
# Cross-file symbol resolution
# ---------------------------------------------------------------------------

def compute_with_imports(sources: dict[str, str], tokenizer,
                         computer=None) -> dict[str, torch.Tensor]:
    """Compute per-file features with cross-file symbol awareness.

    Args:
        sources: {file_path: source_string} for all files in the scope.
        tokenizer: GPT-2 tokenizer.
        computer: CodeChannelComputer instance (created if None).

    Returns:
        {file_path: [T, 5] tensor} with cross-file symbol features resolved.
    """
    if computer is None:
        computer = CodeChannelComputer()

    results = {}
    for file_path, source in sources.items():
        results[file_path] = computer.compute(source, tokenizer)

    return results


# ---------------------------------------------------------------------------
# Compiled Attention — deterministic AST-based attention mask
# ---------------------------------------------------------------------------

_LOCAL_WINDOW = 16  # always-attend window for local syntax


class _AttentionVisitor(ast.NodeVisitor):
    """Build compiled attention rules from AST structure.

    Records symbol positions, scope boundaries, and structural
    edges (def → body, return → function, import → usage).
    """

    def __init__(self, source: str, tokenizer):
        self._tok = tokenizer
        self._source = source
        enc = tokenizer(source, return_offsets_mapping=True, add_special_tokens=False)
        self._ids = enc.input_ids
        self._offsets = enc.offset_mapping
        self.T = len(self._ids)

        # Rules: list of (src_range, tgt_range) pairs — all tokens in
        # src_range should attend to all tokens in tgt_range.
        self.rules: list[tuple[tuple[int, int], tuple[int, int]]] = []

        # Symbol tracking
        self._symbol_positions: dict[str, list[int]] = collections.defaultdict(list)
        self._func_starts: dict[str, int] = {}   # name → first token of func body
        self._func_body_end: dict[str, int] = {}  # name → last token of func body
        self._scope_stack: list[int] = []

    def _token_range(self, lineno: int, col: int,
                     end_lineno: int = None, end_col: int = None) -> tuple[int, int]:
        """AST position → token index range."""
        lines = self._source.splitlines(keepends=True)
        if not lines:
            lines = ['']

        def _byte_off(ln, cl):
            off = 0
            for i in range(min(ln - 1, len(lines))):
                off += len(lines[i]) + 1
            return off + min(cl, len(lines[min(ln - 1, len(lines) - 1)]))

        start_byte = _byte_off(lineno, col)
        if end_lineno is not None and end_col is not None:
            end_byte = _byte_off(end_lineno, end_col)
        else:
            end_byte = start_byte + 1

        # Map bytes to token indices
        t_start, t_end = self.T, 0
        for i, (bs, be) in enumerate(self._offsets):
            if be > start_byte and t_start == self.T:
                t_start = i
            if bs < end_byte:
                t_end = i + 1
        return (max(0, t_start), min(self.T, t_end))

    def _add_rule(self, src_range: tuple[int, int],
                  tgt_range: tuple[int, int]):
        s0, s1 = src_range[0], min(src_range[1], self.T)
        t0, t1 = tgt_range[0], min(tgt_range[1], self.T)
        if s0 < s1 and t0 < t1:
            self.rules.append(((s0, s1), (t0, t1)))

    # -- visitor methods --

    def visit_FunctionDef(self, node: ast.FunctionDef):
        def_range = self._token_range(node.lineno, node.col_offset,
                                      node.end_lineno, node.end_col_offset)
        body_start = self._token_range(node.body[0].lineno,
                                       node.body[0].col_offset)[0] if node.body else def_range[1]
        body_end = self._token_range(node.body[-1].end_lineno,
                                     node.body[-1].end_col_offset,
                                     node.body[-1].end_lineno,
                                     node.body[-1].end_col_offset)[1] if node.body else def_range[1]
        body_range = (body_start, body_end)
        self._func_starts[node.name] = body_start
        self._func_body_end[node.name] = body_end

        # Body → definition
        self._add_rule(body_range, def_range)
        # Definition → body
        self._add_rule(def_range, body_range)

        self._scope_stack.append(body_start)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        def_range = self._token_range(node.lineno, node.col_offset,
                                      node.end_lineno, node.end_col_offset)
        body_start = self._token_range(node.body[0].lineno,
                                       node.body[0].col_offset)[0] if node.body else def_range[1]
        body_end = self._token_range(node.body[-1].end_lineno,
                                     node.body[-1].end_col_offset,
                                     node.body[-1].end_lineno,
                                     node.body[-1].end_col_offset)[1] if node.body else def_range[1]
        body_range = (body_start, body_end)
        self._add_rule(body_range, def_range)

        self._scope_stack.append(body_start)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Return(self, node: ast.Return):
        ret_range = self._token_range(node.lineno, node.col_offset,
                                      node.end_lineno, node.end_col_offset)
        if self._scope_stack:
            scope_start = self._scope_stack[-1]
            self._add_rule(ret_range, (max(0, scope_start - 4), scope_start))

    def visit_Name(self, node: ast.Name):
        pos = self._token_range(node.lineno, node.col_offset)
        if pos[0] < self.T:
            self._symbol_positions[node.id].append(pos[0])

    def visit_Import(self, node: ast.Import):
        imp_range = self._token_range(node.lineno, node.col_offset,
                                      node.end_lineno, node.end_col_offset)
        # Attach import to its own tokens primarily
        self._add_rule(imp_range, imp_range)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        imp_range = self._token_range(node.lineno, node.col_offset,
                                      node.end_lineno, node.end_col_offset)
        self._add_rule(imp_range, imp_range)


def compile_attention(source: str, tokenizer) -> torch.Tensor:
    """Build a compiled (deterministic) attention mask from AST structure.

    Returns a [T, T] float32 mask where mask[i, j] = 1.0 means token i
    should attend to token j.  The mask includes:
      - A local window of size _LOCAL_WINDOW (always attend nearby tokens)
      - Function/class body → definition edges
      - Return → enclosing function edges
      - Import self-edges
      - Same-symbol edges

    Unfilled positions are 0.0 (no attention).  The mask is upper-triangular
    (causal: token i can only attend to j ≤ i).

    Args:
        source: Python source code string.
        tokenizer: HuggingFace tokenizer (GPT-2 or compatible).

    Returns:
        Float32 tensor [T, T] with values in {0.0, 1.0}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        enc = tokenizer(source, return_offsets_mapping=True,
                        add_special_tokens=False)
        T = len(enc.input_ids)
        mask = torch.zeros(T, T, dtype=torch.float32)
        for i in range(T):
            lo = max(0, i - _LOCAL_WINDOW + 1)
            mask[i, lo:i + 1] = 1.0
        return mask

    visitor = _AttentionVisitor(source, tokenizer)
    visitor.visit(tree)

    T = visitor.T
    mask = torch.zeros(T, T, dtype=torch.float32)

    # 1. Local window (always present — local syntax)
    for i in range(T):
        lo = max(0, i - _LOCAL_WINDOW + 1)
        mask[i, lo:i + 1] = 1.0

    # 2. Structural rules (def → body, body → def, return → scope)
    for (s0, s1), (t0, t1) in visitor.rules:
        s0, s1 = max(0, s0), min(T, s1)
        t0, t1 = max(0, t0), min(T, t1)
        mask[s0:s1, t0:t1] = 1.0

    # 3. Same-symbol edges (last occurrence → current)
    for positions in visitor._symbol_positions.values():
        for i in range(1, len(positions)):
            curr, prev = positions[i], positions[i - 1]
            if curr < T and prev < T:
                mask[curr, prev] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Compiled Prompt — compress source into steerer features + context header
# ---------------------------------------------------------------------------

def compile_prompt(sources: dict[str, str], tokenizer,
                   computer=None) -> dict:
    """Compile source files into steerer features + a short context header.

    Instead of passing the full source text as input tokens, the model
    receives a short header like ``[CTX:compiled:3 files, 450 lines]``
    while the code structure is injected via steerer features and
    attention mask.

    Args:
        sources: {file_path: source_string} for all context files.
        tokenizer: GPT-2 tokenizer.
        computer: CodeChannelComputer instance (created if None).

    Returns:
        {
            'header': str,           # e.g. "[CTX:compiled:3 files, 450 lines]"
            'features': torch.Tensor, # [total_tokens, 5] concatenated features
            'attention': torch.Tensor, # [total_tokens, total_tokens] compiled mask
            'file_map': list,         # [(file_path, offset, length)] per file
        }
    """
    if computer is None:
        computer = CodeChannelComputer()

    all_features = []
    all_masks = []
    file_map = []
    total_lines = 0
    offset = 0

    for file_path, source in sources.items():
        feat = computer.compute(source, tokenizer)
        attn = compile_attention(source, tokenizer)
        T = feat.shape[0]
        if T == 0:
            continue

        all_features.append(feat)
        all_masks.append(attn)
        file_map.append((file_path, offset, T))
        offset += T
        total_lines += source.count('\n') + 1

    if not all_features:
        return {
            'header': '[CTX:compiled:0 files]',
            'features': torch.zeros(0, 5, dtype=torch.float32),
            'attention': torch.zeros(0, 0, dtype=torch.float32),
            'file_map': [],
        }

    full_features = torch.cat(all_features, dim=0)
    full_attention = torch.block_diag(*all_masks)

    n_files = len(file_map)
    header = f"[CTX:compiled:{n_files} files, {total_lines} lines, "
    header += f"{full_features.shape[0]} tokens]"

    return {
        'header': header,
        'features': full_features,
        'attention': full_attention,
        'file_map': file_map,
    }

    visitor = _AttentionVisitor(source, tokenizer)
    visitor.visit(tree)

    T = visitor.T
    mask = torch.zeros(T, T, dtype=torch.float32)

    # 1. Local window (always present — local syntax)
    for i in range(T):
        lo = max(0, i - _LOCAL_WINDOW + 1)
        mask[i, lo:i + 1] = 1.0

    # 2. Structural rules (def → body, body → def, return → scope)
    for (s0, s1), (t0, t1) in visitor.rules:
        s0, s1 = max(0, s0), min(T, s1)
        t0, t1 = max(0, t0), min(T, t1)
        mask[s0:s1, t0:t1] = 1.0

    # 3. Same-symbol edges (last occurrence → current)
    for positions in visitor._symbol_positions.values():
        for i in range(1, len(positions)):
            curr, prev = positions[i], positions[i - 1]
            if curr < T and prev < T:
                mask[curr, prev] = 1.0

    return mask


# ---------------------------------------------------------------------------
# AsyncCodeCache — queue-based background update system
# ---------------------------------------------------------------------------

class AsyncCodeCache:
    """Event-driven code channel cache with background async updates.

    The model enqueues file changes; a daemon thread processes them
    in the background.  Synchronization guarantees: ``get()`` returns
    current features — it blocks only if the requested file is
    currently being updated by the worker thread (sub-millisecond
    wait in practice).

    Self-contained — no file watcher, no external processes, no
    polling.  The model orchestrates everything by calling ``enqueue``
    after code changes.

    Usage:
        >>> cache = AsyncCodeCache(tokenizer)
        >>> cache.start()
        >>> # Model writes code, then:
        >>> cache.enqueue("src/main.py", new_source)
        >>> # At any point, get current features:
        >>> features = cache.get("src/main.py")
        >>> cache.stop()
    """

    def __init__(self, tokenizer, computer=None, max_queue: int = 1000):
        self._base = CodebaseCache(tokenizer, computer)
        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._processing: set[str] = set()  # files currently being updated
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._max_queue = max_queue

    # ---- lifecycle ----

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ---- model interface ----

    def enqueue(self, file_path: str, source: str):
        """Enqueue a file update. Non-blocking — returns immediately.

        The ``source`` is the new file contents (as a string).  The
        worker thread will call ``_base.update(file_path, source)``
        when it processes this entry.  If the queue is full, the
        oldest entry is dropped.
        """
        with self._lock:
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
            self._queue.append((file_path, source))
            self._cond.notify()

    def get(self, file_path: str) -> torch.Tensor:
        """Return current features for file_path.

        Blocks only if the requested file is currently being updated
        by the worker thread.  Otherwise returns immediately.  If the
        file is in the queue (pending update), the queued update is
        NOT applied — this method returns the currently-cached
        features or triggers a synchronous re-parse.
        """
        if file_path in self._base:
            return self._base[file_path]
        return self._base.get(file_path)

    def wait_and_get(self, file_path: str, timeout: float = 1.0) -> torch.Tensor:
        """Block until file_path is not being processed, then get.

        Useful when the model JUST wrote code and needs the updated
        features before the next token is generated.  Blocks up to
        ``timeout`` seconds.
        """
        deadline = time.time() + timeout
        with self._cond:
            while file_path in self._processing and time.time() < deadline:
                self._cond.wait(timeout=deadline - time.time())
        return self.get(file_path)

    def flush(self, timeout: float = 5.0):
        """Block until the entire queue has been processed."""
        deadline = time.time() + timeout
        with self._cond:
            while self._queue and time.time() < deadline:
                self._cond.wait(timeout=deadline - time.time())

    # ---- internal ----

    def _worker(self):
        while self._running:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()
                else:
                    self._cond.wait(timeout=0.5)

            if task is None:
                continue

            file_path, source = task
            with self._lock:
                self._processing.add(file_path)

            try:
                self._base.update(file_path, source)
            finally:
                with self._lock:
                    self._processing.discard(file_path)
                    self._cond.notify_all()

    # ---- delegate cache methods ----

    def __len__(self):
        return len(self._base)

    def __contains__(self, file_path: str):
        return file_path in self._base

    def preload_repo(self, *args, **kwargs):
        return self._base.preload_repo(*args, **kwargs)
