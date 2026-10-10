#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: old meta extraction (2x json.loads per data line) vs new
merged extraction (1x). Feeds a recorded SSE file in 1KB chunks, like the
live proxy does.

Usage:  python3 bench_extractor.py [path/to/file.sse]
"""
import importlib.util
import json
import os
import sys
import time

_spec = importlib.util.spec_from_file_location(
    "proxy", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "proxy.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
SSEReasoningExtractor = mod.SSEReasoningExtractor

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "examples", "sample.sse")


def old_style(data: bytes, chunk_size: int):
    """Pre-optimization: extractor + separate per-chunk meta scan."""
    ex = SSEReasoningExtractor()
    meta = {}
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        ex.feed(chunk)
        for line in chunk.split(b"\n"):
            if line.startswith(b"data:"):
                try:
                    o = json.loads(line[5:].strip())
                    if isinstance(o, dict) and o.get("id"):
                        meta = {"id": o["id"], "model": o.get("model"),
                                "created": o.get("created")}
                except Exception:
                    pass
    return meta


def new_style(data: bytes, chunk_size: int):
    """Post-optimization: extractor records meta during the same parse."""
    ex = SSEReasoningExtractor()
    for i in range(0, len(data), chunk_size):
        ex.feed(data[i:i + chunk_size])
    return ex.meta


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    chunk_size = 1024
    with open(path, "rb") as f:
        data = f.read()
    n_data_lines = data.count(b"data:")
    print(f"file: {path}")
    print(f"bytes: {len(data)}, data lines: {n_data_lines}, "
          f"chunk: {chunk_size}")

    runs = 3
    for name, fn in (("old (2x parse)", old_style),
                     ("new (1x parse)", new_style)):
        best = None
        metas = []
        for r in range(runs):
            t0 = time.perf_counter()
            metas.append(fn(data, chunk_size))
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        assert all(m == metas[0] for m in metas), "results differ!"
        print(f"{name}: best of {runs} = {best*1000:.1f} ms")

    # correctness: both must yield the same meta
    m_old = old_style(data, chunk_size)
    m_new = new_style(data, chunk_size)
    print(f"\nmeta identical: {m_old == m_new}")
    print(f"meta: {m_new}")


if __name__ == "__main__":
    main()
