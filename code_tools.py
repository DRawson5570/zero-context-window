"""code_tools.py — Model-callable codebase orchestration tools.

Bridges the compiled-hybrid model's generation loop with the CodebaseCache
and AST analysis. The model emits trigger spans (via the ABI loop in
reasoning_generate.py) like [FILE:...], [SYMBOL:...], [DIAG:...], etc.
These tools resolve them and return results for context injection.

All tools accept primitive-typed arguments (strings, ints) and return
strings — compatible with the executor trigger pattern in executors.py.

Author: DeepSeek v4 Pro · 2026-06-19
"""

from __future__ import annotations

import ast
import os
from typing import Optional

import torch

from code_channel import CodeChannelComputer, CodebaseCache


class CodeTools:
    """Model-callable codebase orchestration.

    Wraps the CodebaseCache and CodeChannelComputer to expose functions
    that the model can invoke via trigger spans during generation.

    Usage:
        >>> tools = CodeTools(cache, tokenizer)
        >>> tools.handle("FILE", "src/main.py")      # file structure summary
        >>> tools.handle("SYMBOL", "authenticate")    # find symbol definition
        >>> tools.handle("DIAG", "src/main.py")       # diagnostics (AST errors)
    """

    def __init__(self, cache, tokenizer):
        self._cache = cache
        self._tok = tokenizer
        self._async = hasattr(cache, 'enqueue')
        self._computer = CodeChannelComputer()

    def handle(self, kind: str, payload: str) -> str:
        """Route a trigger (kind, payload) to the appropriate tool.

        Returns a result string suitable for [RESULT:...] injection.
        On error, returns ERR:<reason> (never raises).
        """
        kind = kind.upper()
        try:
            if kind == "FILE":
                return self._file_structure(payload)
            elif kind == "SYMBOL":
                return self._find_symbol(payload)
            elif kind == "DIAG":
                return self._diagnostics(payload)
            elif kind == "IMPORTS":
                return self._list_imports(payload)
            elif kind == "FEAT":
                return self._feature_summary(payload)
            elif kind == "UPDATE":
                return self._enqueue_update(payload)
            else:
                return f"ERR:unknown tool kind '{kind}'"
        except Exception as e:
            return f"ERR:{type(e).__name__}:{e}"

    # ---- tool implementations ----

    def _file_structure(self, path: str) -> str:
        """Return a structural summary of a file from cached features.

        [FILE:src/main.py] → "function:3, class:1, import:2, lines:45"
        """
        feat = self._cache.get(path)
        if feat is None or feat.numel() == 0:
            source = self._read_or_empty(path)
            if source:
                feat = self._computer.compute(source, self._tok)
            else:
                return f"ERR:file not found: {path}"

        has_func = int((feat[:, 0] > 0.3).sum().item())
        has_class = int((feat[:, 0] > 0.2).sum().item())
        type_hints = int((feat[:, 2] > 0.1).sum().item())
        error_toks = int((feat[:, 4] > 0.5).sum().item())

        source = self._read_or_empty(path)
        line_count = source.count('\n') + 1 if source else 0

        parts = [f"lines:{line_count}"]
        if has_func: parts.append(f"func_def_tokens:{has_func}")
        if has_class: parts.append(f"class_def_tokens:{has_class}")
        if type_hints: parts.append(f"type_hint_tokens:{type_hints}")
        if error_toks: parts.append(f"WARNING:parse_errors:{error_toks}")
        return ", ".join(parts) if parts else f"empty file: {path}"

    def _find_symbol(self, symbol: str) -> str:
        """Search the entire repo cache for a symbol definition.

        [SYMBOL:authenticate] → "src/auth.py:42 (function)"
        """
        import glob as _glob
        symbol = symbol.strip()
        found = []

        for path in list(self._cache._features.keys()):
            source = self._read_or_empty(path)
            if not source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == symbol:
                        found.append(f"{path}:{node.lineno} (function)")
                elif isinstance(node, ast.ClassDef):
                    if node.name == symbol:
                        found.append(f"{path}:{node.lineno} (class)")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == symbol:
                            found.append(f"{path}:{node.lineno} (variable)")
                            break

        if not found:
            return f"symbol '{symbol}' not found in cached repo"
        return ", ".join(found[:5])  # limit to first 5

    def _diagnostics(self, path: str) -> str:
        """Return AST-level diagnostics for a file (no LSP process needed).

        [DIAG:src/main.py] → "OK" or "SyntaxError: invalid syntax at line 42"
        """
        source = self._read_or_empty(path)
        if not source:
            return f"ERR:file not found: {path}"
        try:
            ast.parse(source)
            return "OK (no syntax errors)"
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"

    def _list_imports(self, path: str) -> str:
        """List all imports in a file (stdlib + third-party).

        [IMPORTS:src/main.py] → "os, json, utils.add, typing.List"
        """
        source = self._read_or_empty(path)
        if not source:
            return f"ERR:file not found: {path}"
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return "ERR:parse error"

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ''
                for alias in node.names:
                    imports.append(f"{mod}.{alias.name}")

        return ", ".join(sorted(set(imports))) if imports else "(no imports)"

    def _feature_summary(self, path: str) -> str:
        """Return aggregate feature statistics for a file.

        [FEAT:src/main.py] → "structural:0.12, depth:0.08, type:0.04, symbol:0.01, dist:0.30"
        """
        feat = self._cache.get(path)
        if feat is None or feat.numel() == 0:
            return f"ERR:no features cached for {path}"
        means = feat.mean(dim=0)
        names = ["structural", "depth", "type", "symbol", "distance"]
        return ", ".join(f"{n}:{means[i].item():.2f}" for i, n in enumerate(names))

    def _enqueue_update(self, path: str) -> str:
        """Enqueue file for background reprocessing after model edit.

        [UPDATE:src/main.py] reads current file, enqueues non-blocking.
        Worker thread processes update in background.  With a plain
        CodebaseCache (no async), does a synchronous update.
        """
        source = self._read_or_empty(path)
        if not source:
            return f"ERR:file not found: {path}"
        if self._async:
            self._cache.enqueue(path, source)
            return f"OK:enqueued {path}"
        self._cache.update(path, source)
        return f"OK:updated {path}"

    @staticmethod
    def _read_or_empty(path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
