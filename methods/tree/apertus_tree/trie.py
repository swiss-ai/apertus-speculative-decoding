"""Torch-free pieces of the candidate builder: n-gram lookup and the per-anchor trie."""

from __future__ import annotations

import bisect
from collections import defaultdict

# --------------------------------------------------------------------------- n-gram lookup


class NgramIndex:
    def __init__(self, ids: list[int], max_n: int = 4, min_n: int = 1):
        self.ids, self.max_n, self.min_n = ids, max_n, min_n
        self.pos: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for n in range(min_n, max_n + 1):
            for end in range(n - 1, len(ids)):
                self.pos[tuple(ids[end - n + 1 : end + 1])].append(end)

    def branches(self, a: int, limit: int = 8, depth: int = 7) -> tuple[list[list[int]], int]:
        """Continuations after earlier matches of the suffix ending at ``a``:
        longest n first, most recent first. Returns (branches, longest match n)."""
        out: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        longest = 0
        for n in range(self.max_n, self.min_n - 1, -1):
            if a - n + 1 < 0:
                continue
            ends = self.pos.get(tuple(self.ids[a - n + 1 : a + 1]), [])
            stop = bisect.bisect_left(ends, a)
            if stop and not longest:
                longest = n
            for end in reversed(ends[max(0, stop - 64) : stop]):
                cont = tuple(self.ids[end + 1 : min(end + 1 + depth, a + 1)])
                if cont and cont not in seen:
                    seen.add(cont)
                    out.append(list(cont))
                    if len(out) >= limit:
                        return out, longest
        return out, longest


class Trie:
    """Candidate nodes of one anchor; parents precede children."""

    def __init__(self):
        self.parent: list[int] = []
        self.token: list[int] = []
        self.depth: list[int] = []
        self.flags: list[int] = []
        self.ng: list[int] = []
        self.index: dict[tuple[int, int], int] = {}

    def add_path(self, path: list[int], flag: int, ng: int = -1) -> list[int]:
        parent, out = -1, []
        for depth, tok in enumerate(path, 1):
            key = (parent, tok)
            node = self.index.get(key)
            if node is None:
                node = len(self.token)
                self.index[key] = node
                self.parent.append(parent)
                self.token.append(tok)
                self.depth.append(depth)
                self.flags.append(0)
                self.ng.append(-1)
            self.flags[node] |= flag
            if ng >= 0 and (self.ng[node] < 0 or ng < self.ng[node]):
                self.ng[node] = ng
            out.append(node)
            parent = node
        return out

    def path(self, node: int) -> list[int]:
        out = []
        while node >= 0:
            out.append(self.token[node])
            node = self.parent[node]
        return out[::-1]
