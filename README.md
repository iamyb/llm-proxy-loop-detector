# llm-proxy-loop-detector

A tiny SSE tee proxy with **real-time thinking-loop detection** for local
LLM servers (llama.cpp / llama-swap / vLLM / anything with an
OpenAI-compatible API).

Some models occasionally get stuck repeating the same reasoning paragraph
over and over ("thinking loop"), burning tokens and GPU time for minutes.
This proxy sits in front of the server, watches the
`delta.reasoning_content` stream, and **cuts the stream the moment a loop is
detected** — closing it with a protocol-valid `finish_reason: "stop"` chunk
so the client sees a normal completion.

```
LLM client ──> proxy:9293 ──> LLM server:8080
                 │
                 ├─ tees every request to disk (.req.json + .sse)
                 ├─ detects reasoning loops in real time
                 └─ cuts the stream on trigger (or observe-only mode)
```

## Features

- **Full passthrough** — all headers (incl. `Authorization`) forwarded; SSE
  stream forwarded chunk by chunk with no buffering
- **Loop detection on `reasoning_content` only** — visible `content` is
  never touched; works for both CJK and Latin text (char-level n-gram
  uniqueness ratio over a sliding window)
- **Warmup gate** — detection starts only after N chars of reasoning,
  avoiding false positives on normal thinking preambles
- **Cut or observe** — by default the stream is cut with a valid
  `finish_reason` + `[DONE]` tail; `--no-cut` records the trigger without
  cutting
- **Full request logging** — every request saved as
  `{logdir}/{session}/{ts}_{id}.req.json` + `.sse`; cancelled streams are
  kept too
- **Automatic log cleanup at startup** — rotate `proxy.log` > 10 MB, delete
  session dirs older than `--retain-days`, size cap via `--max-logdir-mb`
- **`/health`** endpoint for liveness probes

## Quick start

### Docker (recommended)

```bash
docker compose up -d --build
curl http://localhost:9293/health   # {"ok":true}
```

Then point your LLM client at `http://localhost:9293` instead of the server
port. The example `docker-compose.yml` assumes the LLM server listens on the
host at `127.0.0.1:8080` (reached via `host.docker.internal`).

### Bare Python

```bash
pip install fastapi uvicorn httpx orjson
python proxy.py --listen 0.0.0.0:9293 --upstream http://127.0.0.1:8080 \
    --logdir ./logs --warmup-chars 2000
```

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--listen` | `0.0.0.0:9293` | proxy bind address |
| `--upstream` | `http://127.0.0.1:8080` | LLM server base URL |
| `--logdir` | `./logs` | where requests are tee'd |
| `--warmup-chars` | `4000` | start detection only after this many reasoning chars |
| `--window` | `3000` | sliding window size (normalized chars) |
| `--ngram` | `18` | char-level n-gram length |
| `--ratio` | `0.5` | unique-gram ratio below this ⇒ loop |
| `--no-cut` | off | observe only, never cut the stream |
| `--client-tag` | *(empty)* | `X-Real-Client` header added upstream |
| `--retain-days` | `7` | delete session dirs older than N days (at startup) |
| `--max-logdir-mb` | `4096` | size cap after age-based cleanup |

## How loop detection works

1. Only `delta.reasoning_content` is fed to the detector (visible content
   is ignored).
2. Nothing happens until the cumulative reasoning length reaches
   `--warmup-chars` (normal thinking preambles are usually fine).
3. The stream is normalized (lowercase, keep only `[0-9a-z]` and CJK
   ideographs) and a sliding window of the last `--window` chars is kept.
4. Every ~100 new chars, the **unique n-gram ratio** inside the window is
   computed: `unique_grams / total_grams`.
   - Normal prose: 18-grams are almost all unique → ratio ≈ 0.95+
   - Repeated paragraph: ratio collapses below `--ratio` (0.5)
5. On trigger (cut mode) the proxy emits:

   ```
   : loop detected: unique_ratio=0.32 (<0.5) top_gram_x4: '...'

   data: {"id": "...", "choices": [{"finish_reason": "stop", ...}]}

   data: [DONE]
   ```

   and stops forwarding. The tail is also written to the `.sse` log file.

## Logs

```
logs/
├── proxy.log              # one JSON line per request (status, bytes,
│                          # duration, reasoning_chars, loop_triggered, hit)
├── proxy.log.1            # rotated previous log
└── {session_id}/          # 12-hex session key (sha1 of first 2 messages)
    ├── 20261009T155148Z_6a72e356.req.json   # request body
    └── 20261009T155148Z_6a72e356.sse        # full SSE stream (+ cut tail)
```

Cleanup runs automatically at startup; `./clean-logs.sh` does it manually.

## Testing

```bash
./run-tests.sh            # 22 regression tests, inside the container
./run-tests.sh -v         # verbose
```

- `test_regression.py` — synthetic, deterministic unit tests for the
  detector, SSE extractor, session key, logging, cleanup, and cut tail
- `test_replay.py` — replays a recorded `.sse` file through the production
  code path (default fixture: `examples/sample.sse`, a synthetic stream
  with a deliberate loop)
- `bench_extractor.py` — extractor throughput benchmark

## License

MIT — see [LICENSE](LICENSE).
