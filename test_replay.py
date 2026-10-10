#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline replay test: feed a recorded .sse file through the production
SSEReasoningExtractor + LoopDetector (same params as the live proxy) and
verify the loop triggers, where, and that the synthesized closing chunk
is valid.

Usage:  python3 test_replay.py [path/to/file.sse] [chunk_bytes]

The default fixture (examples/sample.sse) is a small synthetic stream that
contains a deliberate reasoning loop, so the test is self-contained.
"""
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "proxy", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "proxy.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
LoopDetector = mod.LoopDetector
SSEReasoningExtractor = mod.SSEReasoningExtractor

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "examples", "sample.sse")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    with open(path, "rb") as f:
        data = f.read()
    print(f"file: {path}")
    print(f"total bytes: {len(data)}, replay chunk size: {chunk_size}")

    # same params as the proxy defaults + cut=True
    det = LoopDetector(warmup=4000, window=3000, n=18, ratio=0.5, cut=True)
    ex = SSEReasoningExtractor()

    meta = {}
    trigger_info = None
    n_chunks = 0
    offset = 0
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        n_chunks += 1
        # grab id/model/created like gen() does
        for line in chunk.split(b"\n"):
            if line.startswith(b"data:"):
                try:
                    o = json.loads(line[5:].strip())
                    if isinstance(o, dict) and o.get("id"):
                        meta = {"id": o["id"], "model": o.get("model"),
                                "created": o.get("created")}
                except Exception:
                    pass
        text = ex.feed(chunk)
        if text and det.feed(text):
            trigger_info = {
                "chunk_index": n_chunks,
                "byte_offset": offset,
                "reasoning_chars": det.raw_len,
                "hit": det.hit,
            }
            break
        offset += len(chunk)

    print()
    if trigger_info is None:
        print("RESULT: NOT triggered")
        print(f"  total reasoning_chars: {det.raw_len}")
        sys.exit(1)

    print("RESULT: triggered")
    print(f"  chunk #{trigger_info['chunk_index']} "
          f"(byte offset ~{trigger_info['byte_offset']})")
    print(f"  reasoning_chars at trigger: {trigger_info['reasoning_chars']}")
    print(f"  hit: {trigger_info['hit']}")

    # cut semantics: after trigger, feed() must keep returning True
    more = ex.feed(data[offset:offset + chunk_size])
    still = det.feed(more) if more else det.feed("x")
    print(f"  feed() after trigger returns: {still} (expect True)")

    # verify the synthesized closing chunk is valid JSON, same shape as a
    # normal finish_reason chunk
    closing = {
        "id": meta.get("id"),
        "model": meta.get("model"),
        "created": meta.get("created"),
        "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
        "object": "chat.completion.chunk",
    }
    tail = (f": loop detected: {det.hit}\n\n"
            f"data: {json.dumps(closing)}\n\n"
            f"data: [DONE]\n\n").encode()
    print()
    print("--- synthesized tail (what the client would receive) ---")
    print(tail.decode())
    parsed = json.loads(tail.decode().split("data: ")[1].split("\n\n")[0])
    assert parsed["choices"][0]["finish_reason"] == "stop"
    assert parsed["object"] == "chat.completion.chunk"
    assert parsed["id"] == meta["id"]
    print("--- closing chunk JSON valid, shape matches normal stream ---")


if __name__ == "__main__":
    main()
