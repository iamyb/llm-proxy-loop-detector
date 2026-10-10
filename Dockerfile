FROM python:3.12-slim

WORKDIR /app

# Aliyun mirror for faster builds in CN; remove the -i flag elsewhere
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn httpx orjson

COPY proxy.py test_regression.py test_replay.py /app/
COPY examples/ /app/examples/

# NOTE: no VOLUME declaration here — a VOLUME /app/logs would make Docker
# attach a named volume whenever the compose file bind-mounts the parent
# /app instead of /app/logs exactly, silently hiding the host logs dir.
# Mounting is left to docker-compose.yml.

EXPOSE 9293

CMD ["python", "proxy.py"]
