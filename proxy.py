#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm-proxy-loop-detector — SSE tee + thinking-loop detector in front of a
local LLM server (OpenAI-compatible API).

Topology:  LLM client -> proxy:9293 -> LLM server (llama.cpp / llama-swap /
           vLLM / ...)

Features:
  1. Full passthrough (all headers incl. Authorization); SSE stream is
     forwarded chunk by chunk with no buffering
  2. Every request is persisted: {logdir}/{session}/{ts}_{id}.req.json + .sse
     (content of cancelled streams is kept as well)
  3. Real-time loop detection on delta.reasoning_content only, starting
     after the cumulative length reaches --warmup-chars; criterion is the
     unique-ngram ratio inside a sliding window (works for CJK and Latin)
  4. On trigger the stream is cut by default (--no-cut = observe only),
     closed with a protocol-valid finish_reason chunk + [DONE]

Deps:   pip install fastapi uvicorn httpx orjson
Run:    python proxy.py --listen 0.0.0.0:9293 --upstream http://127.0.0.1:8080 \
             --logdir /path/to/logs --warmup-chars 2000
"""
import argparse
import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

ARGS = None
client: httpx.AsyncClient = None

app = FastAPI()


# ---------------------------------------------------------------- detector

class LoopDetector:
    """Real-time loop detection on reasoning_content.

    warmup:   no counting until the cumulative raw reasoning length reaches
              this value (avoids false positives on normal thinking preamble,
              and saves CPU)
    window:   the criterion only looks at the most recent N normalized chars
    n:        char-level n-gram length (CJK has no spaces, so char-level is
              required)
    ratio:    unique_grams / total_grams inside the window below this value
              => loop. Normal text 18-grams are almost all unique
              (ratio~0.95+); repeated paragraphs drop below 0.5
    """

    _CLEAN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")

    def __init__(self, warmup=4000, window=3000, n=18, ratio=0.5, cut=True):
        self.warmup = warmup
        self.window = window
        self.n = n
        self.ratio = ratio
        self.cut = cut            # False = observe only, never cut
        self.raw_len = 0          # cumulative raw reasoning length (warmup)
        self.norm = ""            # normalized char stream (capped at 20k)
        self.pending = 0
        self.triggered = False
        self.hit = None

    def feed(self, text: str) -> bool:
        """Feed a reasoning_content increment; True means the stream should
        be cut.

        With cut=False (observation mode) always returns False, but
        triggered/hit are still recorded."""
        if self.triggered:
            return self.cut
        self.raw_len += len(text)
        if self.raw_len < self.warmup:      # warmup gate: no detection yet
            return False
        self.norm = (self.norm + self._CLEAN.sub("", text.lower()))[-20000:]
        self.pending += len(text)
        if self.pending < 100:              # re-check every ~100 chars, save CPU
            return False
        self.pending = 0
        return self._check()

    def _check(self) -> bool:
        w = self.norm[-self.window:]
        if len(w) < self.window // 2:
            return False
        n = self.n
        grams = [w[i:i + n] for i in range(len(w) - n + 1)]
        total = len(grams)
        uniq = len(set(grams))
        ratio = uniq / total
        if ratio < self.ratio:
            c, g = Counter(grams).most_common(1)[0]
            self.triggered = True
            self.hit = (f"unique_ratio={ratio:.2f} (<{self.ratio}) "
                        f"top_gram_x{c}: {g!r}")
            return self.cut
        return False


class SSEReasoningExtractor:
    """Incrementally extract delta.reasoning_content from the SSE byte stream.

    Also records id/model/created from the latest parsed chunk in .meta so
    the proxy can synthesize a protocol-valid closing chunk on cut without
    re-parsing the same data lines."""

    def __init__(self):
        self.buf = b""
        self.meta = {}

    def feed(self, data: bytes) -> str:
        self.buf += data
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = orjson.loads(payload)   # bytes in, faster than json
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("id"):
                self.meta = {"id": obj["id"],
                             "model": obj.get("model"),
                             "created": obj.get("created")}
            try:
                delta = obj["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            t = delta.get("reasoning_content")
            if isinstance(t, str) and t:
                out.append(t)
        return "".join(out)


def build_cut_tail(det: "LoopDetector", meta: dict) -> bytes:
    """Build the protocol-valid tail emitted when a loop is cut:
    one SSE comment line (ignored by clients), a content chunk carrying a
    visible notice (so clients that got no answer text don't show
    "no response"), a finish_reason chunk (id/model/created from meta,
    recorded during the same parse), and [DONE]. The client sees a normal
    completion."""
    notice = ("[llm-proxy] response cut: model entered a repetition loop "
              f"({det.hit})")
    notice_chunk = {
        "id": meta.get("id"),
        "model": meta.get("model"),
        "created": meta.get("created"),
        "choices": [{"finish_reason": None, "index": 0,
                     "delta": {"content": notice}}],
        "object": "chat.completion.chunk",
    }
    closing = {
        "id": meta.get("id"),
        "model": meta.get("model"),
        "created": meta.get("created"),
        "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
        "object": "chat.completion.chunk",
    }
    return (f": loop detected: {det.hit}\n\n"
            f"data: {orjson.dumps(notice_chunk).decode()}\n\n"
            f"data: {orjson.dumps(closing).decode()}\n\n"
            f"data: [DONE]\n\n").encode()


# ---------------------------------------------------------------- session id

def session_key(body: bytes) -> str:
    """Derive a session id: many LLM clients carry no session id, but the
    system prompt + first user message are constant within a session (later
    turns only append messages), so the first 12 hex chars of their sha1
    serve as a stable key. Non-chat / parse failure -> 'nosession'."""
    try:
        msgs = orjson.loads(body).get("messages") or []
        parts = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS)
                 for m in msgs[:2]]
        if not parts:
            return "nosession"
        return hashlib.sha1(b"\n".join(parts)).hexdigest()[:12]
    except Exception:
        return "nosession"


# ---------------------------------------------------------------- logging

def log_summary(rec: dict):
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    # orjson emits UTF-8 as-is (no \uXXXX escaping) by default
    with open(os.path.join(ARGS.logdir, "proxy.log"), "ab") as f:
        f.write(orjson.dumps(rec) + b"\n")


_SESS_CACHE = {}


def session_dir(sess: str) -> str:
    """On-disk session directory: {first_seen_ts}_{key}, created on first
    use. The timestamp makes sessions sortable by age at a glance; the
    key suffix lets us find the same session again after a restart (scan
    for an existing dir ending in _{key}), so all turns of a session stay
    in one directory. 'nosession' stays a plain, untimestamped dir."""
    if sess == "nosession":
        d = os.path.join(ARGS.logdir, "nosession")
        os.makedirs(d, exist_ok=True)
        return d
    cached = _SESS_CACHE.get(sess)
    if cached:
        return cached
    try:
        for name in os.listdir(ARGS.logdir):
            # new format: {ts}_{key}; legacy format (pre-timestamp): {key}
            if name.endswith("_" + sess) or name == sess:
                d = os.path.join(ARGS.logdir, name)
                if os.path.isdir(d):
                    _SESS_CACHE[sess] = d
                    return d
    except OSError:
        pass
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = os.path.join(ARGS.logdir, f"{ts}_{sess}")
    os.makedirs(d, exist_ok=True)
    _SESS_CACHE[sess] = d
    return d


def save_request(req_id: str, sess_dir: str, path: str, body: bytes):
    p = os.path.join(sess_dir, f"{req_id}.req.json")
    try:
        obj = orjson.loads(body)
        with open(p, "wb") as f:
            f.write(orjson.dumps({"path": path, "body": obj}))
    except Exception:
        with open(p, "wb") as f:
            f.write(body)


# ---------------------------------------------------------------- log cleanup

def _dir_size_mb(d: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total // (1024 * 1024)


def cleanup_logs():
    """Run at startup (machines that are only powered on intermittently have
    no cron, so startup is the natural cleanup point):
    1. rotate proxy.log if it exceeds 10MB (keep one .1 backup)
    2. delete session dirs older than --retain-days
    3. if total still exceeds --max-logdir-mb, delete oldest dirs until below
    """
    logdir = ARGS.logdir
    # 1. summary log rotation
    logp = os.path.join(logdir, "proxy.log")
    try:
        if os.path.exists(logp) and os.path.getsize(logp) > 10 * 1024 * 1024:
            os.replace(logp, logp + ".1")
            print(f"[cleanup] rotated proxy.log -> proxy.log.1",
                  flush=True)
    except OSError as e:
        print(f"[cleanup] log rotation failed: {e!r}", flush=True)

    # collect session dirs (oldest first by mtime)
    dirs = []
    for name in os.listdir(logdir):
        d = os.path.join(logdir, name)
        if os.path.isdir(d):
            try:
                dirs.append((os.path.getmtime(d), d))
            except OSError:
                pass
    dirs.sort()

    # 2. age-based
    cutoff = time.time() - ARGS.retain_days * 86400
    for mtime, d in list(dirs):
        if mtime < cutoff:
            sz = _dir_size_mb(d)
            shutil.rmtree(d, ignore_errors=True)
            dirs.remove((mtime, d))
            print(f"[cleanup] removed {os.path.basename(d)} "
                  f"({sz}MB, older than {ARGS.retain_days}d)", flush=True)

    # 3. size-based fallback
    total = sum(_dir_size_mb(d) for _m, d in dirs)
    for mtime, d in dirs:
        if total <= ARGS.max_logdir_mb:
            break
        sz = _dir_size_mb(d)
        shutil.rmtree(d, ignore_errors=True)
        total -= sz
        print(f"[cleanup] removed {os.path.basename(d)} ({sz}MB, over "
              f"{ARGS.max_logdir_mb}MB cap)", flush=True)
    print(f"[cleanup] logdir now ~{total}MB across {len(dirs)} sessions",
          flush=True)


# ---------------------------------------------------------------- proxy

@app.get("/health")
async def health():
    """Liveness probe (registered before the catch-all so it wins)."""
    return {"ok": True}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    body = await request.body()
    req_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    if ARGS.client_tag:
        headers["X-Real-Client"] = ARGS.client_tag
    url = ARGS.upstream.rstrip("/") + "/" + path
    if request.url.query:
        url += "?" + request.url.query

    is_chat = path.endswith("/chat/completions") or path.endswith("/completions")
    sess = session_key(body) if is_chat else "nosession"
    sess_dir = await asyncio.to_thread(session_dir, sess)
    # offload the (potentially large) request-body write off the event loop
    await asyncio.to_thread(save_request, req_id, sess_dir, path, body)

    t0 = time.time()
    up_req = client.build_request(request.method, url, content=body, headers=headers)
    try:
        upstream = await client.send(up_req, stream=True)
    except Exception as e:
        # upstream unreachable (e.g. LLM server restarting): log a summary, then 502
        await asyncio.to_thread(log_summary, {
            "id": req_id, "session": sess, "path": path,
            "status": "upstream_error", "error": repr(e),
            "duration_s": round(time.time() - t0, 2),
        })
        raise HTTPException(status_code=502, detail=f"upstream error: {e!r}")

    det = LoopDetector(warmup=ARGS.warmup_chars, window=ARGS.window,
                       n=ARGS.ngram, ratio=ARGS.ratio,
                       cut=not ARGS.no_cut) if is_chat else None
    extractor = SSEReasoningExtractor() if is_chat else None

    # Forward only safe response headers; Content-Encoding/Length are stale
    # after httpx decoding and must be dropped
    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in ("content-encoding", "content-length",
                                         "transfer-encoding")}

    async def gen():
        n_bytes = 0
        sse_file = None
        # batched flush: one flush() per chunk costs a syscall per token;
        # flush only when 32KB accumulated or 0.5s passed (whichever first)
        FLUSH_BYTES = 32 * 1024
        FLUSH_INTERVAL = 0.5
        unflushed = 0
        last_flush = time.monotonic()
        try:
            # open inside try: finally closes the handle even if gen() is
            # never iterated or cancelled early
            sse_file = open(os.path.join(sess_dir, f"{req_id}.sse"), "wb")
            async for chunk in upstream.aiter_bytes():
                sse_file.write(chunk)
                n_bytes += len(chunk)
                unflushed += len(chunk)
                if unflushed >= FLUSH_BYTES or \
                        time.monotonic() - last_flush >= FLUSH_INTERVAL:
                    sse_file.flush()
                    unflushed = 0
                    last_flush = time.monotonic()
                if det is not None:
                    text = extractor.feed(chunk)
                    if text and det.feed(text):
                        tail = build_cut_tail(det, extractor.meta)
                        sse_file.write(tail)
                        n_bytes += len(tail)
                        yield tail
                        break
                yield chunk
        finally:
            if sse_file is not None:
                sse_file.flush()   # make sure the tail window is persisted
                sse_file.close()
            await upstream.aclose()
            await asyncio.to_thread(log_summary, {
                "id": req_id, "session": sess, "path": path,
                "status": upstream.status_code,
                "bytes": n_bytes,
                "duration_s": round(time.time() - t0, 2),
                "reasoning_chars": det.raw_len if det else None,
                "loop_triggered": bool(det and det.triggered),
                "hit": det.hit if det else None,
            })

    return StreamingResponse(gen(), status_code=upstream.status_code,
                             headers=resp_headers)


@asynccontextmanager
async def lifespan(_app):
    global client
    os.makedirs(ARGS.logdir, exist_ok=True)
    cleanup_logs()
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    try:
        yield
    finally:
        await client.aclose()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:9293")
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--logdir", default="./logs")
    ap.add_argument("--warmup-chars", type=int, default=4000,
                    help="start detection only after reasoning_content exceeds this many chars")
    ap.add_argument("--window", type=int, default=3000,
                    help="criterion window: only the most recent N normalized chars")
    ap.add_argument("--ngram", type=int, default=18,
                    help="char-level n-gram length")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="unique_grams/total_grams below this value => loop")
    ap.add_argument("--no-cut", action="store_true",
                    help="observe only, never cut the stream")
    ap.add_argument("--client-tag", default="",
                    help="X-Real-Client header passed upstream to identify the source in activity logs")
    ap.add_argument("--retain-days", type=int, default=7,
                    help="delete session dirs whose last activity is older than N days (at startup)")
    ap.add_argument("--max-logdir-mb", type=int, default=4096,
                    help="after age-based cleanup, delete oldest session dirs until total is below N MB")
    ARGS = ap.parse_args()

    host, port = ARGS.listen.rsplit(":", 1)
    uvicorn.run(app, host=host, port=int(port), log_level="warning")
