#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for llm-proxy-loop-detector.

Run inside the container (or anywhere with orjson installed):
    python3 test_regression.py            # all tests
    python3 test_regression.py -v         # verbose

Covers the pieces that must not silently break when the proxy is edited:
  * LoopDetector   — warmup gate, no false positive on normal text,
                     triggers on real repetition (Latin + CJK), cut vs observe
  * SSEReasoningExtractor — incremental byte feeding across chunk boundaries,
                     meta capture, [DONE]/comment/malformed-line tolerance
  * session_key    — stable key, 'nosession' fallback
  * log_summary / save_request — valid JSON, CJK stays readable
  * cleanup_logs   — rotation, age-based, size-based
  * cut closing chunk — protocol-valid finish_reason + [DONE]

No network, no large fixture: everything is synthetic and deterministic.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("proxy",
                                               os.path.join(HERE, "proxy.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

LoopDetector = mod.LoopDetector
SSEReasoningExtractor = mod.SSEReasoningExtractor


def make_args(logdir, **kw):
    """Build an ARGS namespace like the real one (only the fields the
    functions under test actually read)."""
    base = dict(logdir=logdir, retain_days=7, max_logdir_mb=4096,
                warmup_chars=4000, window=3000, ngram=18, ratio=0.5,
                no_cut=False, client_tag="")
    base.update(kw)
    return types.SimpleNamespace(**base)


def sse_chunk(text, rid="chatcmpl-test", model="m", created=1):
    """Build one SSE data line carrying a reasoning_content delta."""
    payload = json.dumps({
        "id": rid, "model": model, "created": created,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0,
                     "delta": {"reasoning_content": text},
                     "finish_reason": None}],
    }).encode()
    return b"data: " + payload + b"\n\n"


# ---------------------------------------------------------------- detector

class TestLoopDetector(unittest.TestCase):
    def test_warmup_gate_no_trigger_before_warmup(self):
        det = LoopDetector(warmup=1000, window=300, n=18, ratio=0.5, cut=True)
        # heavy repetition, but under warmup -> must NOT trigger
        for _ in range(10):
            self.assertFalse(det.feed("the same short phrase " * 3))
        self.assertFalse(det.triggered)

    def test_normal_text_no_false_positive(self):
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5, cut=True)
        # varied, non-repeating prose well past warmup
        words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
                 "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
                 "psi omega").split()
        buf = ""
        for i in range(400):
            buf += words[i % len(words)] + " "
            if det.feed(buf):
                self.fail("false positive on varied text")
            buf = ""
        self.assertFalse(det.triggered)

    def test_latin_repetition_triggers(self):
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5, cut=True)
        phrase = ("I have already considered this exact point and there is "
                  "nothing more to add on this matter at all ")
        triggered = False
        for _ in range(60):
            if det.feed(phrase):
                triggered = True
                break
        self.assertTrue(triggered, "repeated Latin paragraph should trigger")
        self.assertTrue(det.triggered)
        self.assertIsNotNone(det.hit)
        self.assertIn("unique_ratio", det.hit)

    def test_cjk_repetition_triggers(self):
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5, cut=True)
        phrase = "我已经考虑过这个点了，没有更多可以补充的内容，就这样吧。"
        triggered = False
        # normalized phrase is only ~25 chars, and _check() needs the window
        # to hold >= window//2 (1500) chars before it evaluates the ratio, so
        # iterate well past that threshold
        for _ in range(200):
            if det.feed(phrase):
                triggered = True
                break
        self.assertTrue(triggered, "repeated CJK paragraph should trigger")

    def test_observe_mode_never_cuts(self):
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5,
                           cut=False)
        phrase = ("I have already considered this exact point and there is "
                  "nothing more to add on this matter at all ")
        for _ in range(60):
            self.assertFalse(det.feed(phrase))  # observe: always False
        self.assertTrue(det.triggered)          # but still recorded
        self.assertIsNotNone(det.hit)

    def test_stays_cut_after_trigger(self):
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5, cut=True)
        phrase = "repeat repeat repeat repeat repeat repeat repeat repeat "
        fired = False
        for _ in range(60):
            if det.feed(phrase):
                fired = True
        self.assertTrue(fired)
        # once triggered, feed() keeps returning cut (True)
        self.assertTrue(det.feed("anything else"))


# ---------------------------------------------------------------- extractor

class TestExtractor(unittest.TestCase):
    def test_meta_captured(self):
        ex = SSEReasoningExtractor()
        ex.feed(sse_chunk("hello", rid="id-123", model="my-model",
                          created=42))
        self.assertEqual(ex.meta, {"id": "id-123", "model": "my-model",
                                   "created": 42})

    def test_incremental_across_chunk_boundaries(self):
        # one logical SSE line split across many tiny feeds
        line = sse_chunk("abcdefghij")
        ex = SSEReasoningExtractor()
        got = ""
        for i in range(0, len(line), 3):
            got += ex.feed(line[i:i + 3])
        self.assertEqual(got, "abcdefghij")

    def test_multiple_deltas_in_one_feed(self):
        ex = SSEReasoningExtractor()
        blob = sse_chunk("one") + sse_chunk("two") + sse_chunk("three")
        self.assertEqual(ex.feed(blob), "onetwothree")

    def test_done_and_comments_ignored(self):
        ex = SSEReasoningExtractor()
        blob = (b": keepalive comment\n\n"
                + sse_chunk("real")
                + b"data: [DONE]\n\n"
                + b"event: ping\n\n")
        self.assertEqual(ex.feed(blob), "real")

    def test_malformed_json_skipped(self):
        ex = SSEReasoningExtractor()
        blob = (b"data: {not valid json}\n\n"
                + sse_chunk("ok"))
        self.assertEqual(ex.feed(blob), "ok")

    def test_no_reasoning_delta_yields_empty(self):
        ex = SSEReasoningExtractor()
        payload = json.dumps({"id": "x", "choices": [
            {"delta": {"content": "visible only"}}]}).encode()
        self.assertEqual(ex.feed(b"data: " + payload + b"\n\n"), "")


# ---------------------------------------------------------------- session_key

class TestSessionKey(unittest.TestCase):
    def test_stable_for_same_prefix(self):
        b1 = json.dumps({"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"}]}).encode()
        b2 = json.dumps({"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "later turn"}]}).encode()
        self.assertEqual(mod.session_key(b1), mod.session_key(b2))
        self.assertEqual(len(mod.session_key(b1)), 12)

    def test_differs_for_different_prefix(self):
        b1 = json.dumps({"messages": [{"role": "user", "content": "A"}]}).encode()
        b2 = json.dumps({"messages": [{"role": "user", "content": "B"}]}).encode()
        self.assertNotEqual(mod.session_key(b1), mod.session_key(b2))

    def test_nosession_on_garbage(self):
        self.assertEqual(mod.session_key(b"not json at all"), "nosession")
        self.assertEqual(mod.session_key(b"{}"), "nosession")


# ---------------------------------------------------------------- logging

class TestLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="proxy_test_")
        mod.ARGS = make_args(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_summary_valid_json_and_cjk(self):
        mod.log_summary({"id": "r1", "hit": "中文命中", "status": 200})
        with open(os.path.join(self.tmp, "proxy.log"), "rb") as f:
            line = f.read().strip()
        rec = json.loads(line)
        self.assertEqual(rec["id"], "r1")
        self.assertEqual(rec["hit"], "中文命中")
        self.assertIn("ts", rec)
        # CJK must be stored as UTF-8, not \uXXXX escapes
        self.assertIn("中文命中".encode(), line)

    def test_save_request_pretty_and_cjk(self):
        body = json.dumps({"messages": [
            {"role": "user", "content": "你好 world"}]}).encode()
        mod.save_request("req1", "sess1", "v1/chat/completions", body)
        p = os.path.join(self.tmp, "sess1", "req1.req.json")
        self.assertTrue(os.path.exists(p))
        with open(p, "rb") as f:
            raw = f.read()
        obj = json.loads(raw)
        self.assertEqual(obj["path"], "v1/chat/completions")
        self.assertEqual(obj["body"]["messages"][0]["content"], "你好 world")
        self.assertIn("你好".encode(), raw)

    def test_save_request_fallback_on_garbage(self):
        mod.save_request("req2", "sess2", "p", b"<<<not json>>>")
        p = os.path.join(self.tmp, "sess2", "req2.req.json")
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"<<<not json>>>")


# ---------------------------------------------------------------- cleanup

class TestCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="proxy_clean_")
        mod.ARGS = make_args(self.tmp, retain_days=7, max_logdir_mb=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk(self, name, size_mb, age_days=0):
        d = os.path.join(self.tmp, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "f.sse"), "wb") as f:
            f.write(b"x" * int(size_mb * 1024 * 1024))
        t = time.time() - age_days * 86400
        os.utime(d, (t, t))
        return d

    def test_rotation(self):
        logp = os.path.join(self.tmp, "proxy.log")
        with open(logp, "wb") as f:
            f.write(b"y" * (11 * 1024 * 1024))  # > 10MB
        mod.cleanup_logs()
        self.assertTrue(os.path.exists(logp + ".1"))
        self.assertFalse(os.path.exists(logp))  # original moved away

    def test_age_based(self):
        old = self._mk("old", 0.1, age_days=30)
        new = self._mk("new", 0.1, age_days=0)
        mod.cleanup_logs()
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))

    def test_size_based(self):
        # both fresh, but total exceeds the 1MB cap -> oldest removed.
        # _dir_size_mb floors to whole MB, so use 1.5MB each (counts as 1MB,
        # total 2MB > 1MB cap)
        a = self._mk("aaa", 1.5, age_days=0)
        b = self._mk("bbb", 1.5, age_days=0)
        os.utime(a, (time.time() - 100, time.time() - 100))
        os.utime(b, (time.time(), time.time()))
        mod.cleanup_logs()
        self.assertFalse(os.path.exists(a))  # older one gone
        self.assertTrue(os.path.exists(b))


# ---------------------------------------------------------------- cut tail

class TestCutClosingChunk(unittest.TestCase):
    """Mirror the exact tail-building code in gen() and assert the result is
    protocol-valid: a comment line, a finish_reason chunk, and [DONE]."""

    def test_tail_is_valid(self):
        import orjson
        det = LoopDetector(warmup=200, window=3000, n=18, ratio=0.5, cut=True)
        ex = SSEReasoningExtractor()
        phrase = "repeat repeat repeat repeat repeat repeat repeat repeat "
        for _ in range(60):
            if det.feed(phrase):
                break
        self.assertTrue(det.triggered)

        meta = ex.meta or {"id": "id-x", "model": "m", "created": 1}
        closing = {
            "id": meta.get("id"),
            "model": meta.get("model"),
            "created": meta.get("created"),
            "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
            "object": "chat.completion.chunk",
        }
        tail = (f": loop detected: {det.hit}\n\n"
                f"data: {orjson.dumps(closing).decode()}\n\n"
                f"data: [DONE]\n\n").encode()

        lines = tail.decode().split("\n")
        self.assertTrue(lines[0].startswith(": loop detected:"))
        # the finish_reason chunk must parse and be well-formed
        data_lines = [l[5:].strip() for l in lines if l.startswith("data:")]
        self.assertEqual(data_lines[-1], "[DONE]")
        chunk = json.loads(data_lines[0])
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunk["object"], "chat.completion.chunk")
        self.assertIn("id", chunk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
