FROM python:3.12-slim

WORKDIR /app

# Aliyun mirror for faster builds in CN; remove the -i flag elsewhere
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn httpx orjson

COPY proxy.py test_regression.py test_replay.py bench_extractor.py /app/
COPY examples/ /app/examples/

# logs are expected on a volume at /app/logs (see docker-compose.yml)
VOLUME ["/app/logs"]

EXPOSE 9293

CMD ["python", "proxy.py"]
