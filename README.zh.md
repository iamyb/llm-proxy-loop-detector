# llm-proxy-loop-detector

> 中文文档 | [English](README.md)

一个带**实时思考循环检测**的 SSE 透传代理，用于本地 LLM 服务器
（llama.cpp / llama-swap / vLLM / 任何 OpenAI 兼容 API）。

有些模型偶尔会陷入"思考死循环"——反复输出同一段 reasoning 内容，
白白浪费几分钟的 token 和 GPU 时间。这个代理部署在服务器前面，
实时监控 `delta.reasoning_content` 流，**一旦检测到循环立即切断流**，
并用协议合法的 `finish_reason: "stop"` chunk 收尾，客户端看到的是一次
正常完成的响应。

```
LLM 客户端 ──> proxy:9293 ──> LLM 服务器:8080
                 │
                 ├─ 每个请求完整落盘（.req.json + .sse）
                 ├─ 实时检测 reasoning 循环
                 └─ 触发时切断流（或仅观察模式）
```

## 特性

- **完全透传** —— 所有 header（含 `Authorization`）原样转发；SSE 流
  逐 chunk 转发，零缓冲
- **只检测 `reasoning_content`** —— 可见的 `content` 完全不受影响；
  中英文都适用（字符级 n-gram 唯一率 + 滑动窗口）
- **预热门控** —— reasoning 累计超过 N 字符后才开始检测，避免对正常
  思考前缀误报
- **切断或观察** —— 默认触发时切断流，并补发合法的 `finish_reason` +
  `[DONE]` 收尾；`--no-cut` 只记录不切断
- **完整请求日志** —— 每个请求保存为
  `{logdir}/{session}/{ts}_{id}.req.json` + `.sse`；客户端中途取消的流也完整保留
- **启动时自动清理日志** —— `proxy.log` 超 10MB 轮转，按
  `--retain-days` 删除过期会话目录，`--max-logdir-mb` 限制总大小
- **`/health`** 健康检查端点

## 快速开始

### Docker（推荐）

```bash
docker compose up -d --build
curl http://localhost:9293/health   # {"ok":true}
```

然后把 LLM 客户端指向 `http://localhost:9293` 而不是服务器端口。
示例 `docker-compose.yml` 假设 LLM 服务器运行在宿主机
`127.0.0.1:8080`（容器内通过 `host.docker.internal` 访问）。

### 直接运行（不用 Docker）

```bash
pip install fastapi uvicorn httpx orjson
python proxy.py --listen 0.0.0.0:9293 --upstream http://127.0.0.1:8080 \
    --logdir ./logs --warmup-chars 2000
```

## 配置项

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--listen` | `0.0.0.0:9293` | 代理监听地址 |
| `--upstream` | `http://127.0.0.1:8080` | LLM 服务器地址 |
| `--logdir` | `./logs` | 请求日志保存目录 |
| `--warmup-chars` | `4000` | reasoning 超过该字符数后才开始检测 |
| `--window` | `3000` | 滑动窗口大小（归一化后字符数） |
| `--ngram` | `18` | 字符级 n-gram 长度 |
| `--ratio` | `0.5` | 唯一 n-gram 比率低于该值 ⇒ 判定循环 |
| `--no-cut` | 关 | 仅观察，不切断流 |
| `--client-tag` | *(空)* | 向上游添加的 `X-Real-Client` header |
| `--retain-days` | `7` | 启动时删除超过 N 天的会话目录 |
| `--max-logdir-mb` | `4096` | 日志目录总大小上限（按文件新旧清理） |

## 循环检测原理

1. 只有 `delta.reasoning_content` 会喂给检测器（可见 content 忽略）。
2. 累计 reasoning 长度达到 `--warmup-chars` 之前不做任何检测
   （正常的思考前缀通常没问题）。
3. 文本先做归一化（转小写，只保留 `[0-9a-z]` 和 CJK 汉字），维护最近
   `--window` 个字符的滑动窗口。
4. 每新增约 100 个字符，计算窗口内的**唯一 n-gram 比率**：
   `unique_grams / total_grams`。
   - 正常文本：18-gram 几乎全唯一 → 比率 ≈ 0.95+
   - 重复段落：比率骤降到 `--ratio`（0.5）以下
5. 触发后（切断模式）代理会发出：

   ```
   : loop detected: unique_ratio=0.32 (<0.5) top_gram_x4: '...'

   data: {"id": "...", "choices": [{"finish_reason": "stop", ...}]}

   data: [DONE]
   ```

   然后停止转发。这个收尾尾部同样会写入 `.sse` 日志文件。

## 日志

```
logs/
├── proxy.log              # 每个请求一行 JSON（status、bytes、
│                          # duration、reasoning_chars、loop_triggered、hit）
├── proxy.log.1            # 轮转出的上一份日志
└── {session_id}/          # 12 位十六进制会话键（前 2 条消息的 sha1）
    ├── 20261009T155148Z_6a72e356.req.json   # 请求体
    └── 20261009T155148Z_6a72e356.sse        # 完整 SSE 流（含切断尾部）
```

清理在每次启动时自动执行；也可以随时运行 `./clean-logs.sh` 手动清理。

## 测试

```bash
./run-tests.sh            # 22 个回归测试，在容器内运行
./run-tests.sh -v         # 详细输出
```

- `test_regression.py` —— 针对检测器、SSE 解析器、会话键、日志、清理、
  切断尾部的合成数据单元测试（确定性、无网络）
- `test_replay.py` —— 用生产代码路径回放录制的 `.sse` 文件
  （默认 fixture：`examples/sample.sse`，含刻意构造的循环）
- `bench_extractor.py` —— SSE 解析器吞吐基准测试

## 许可

MIT — 见 [LICENSE](LICENSE)。
