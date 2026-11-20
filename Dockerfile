FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -e .
COPY configs ./configs
COPY app ./app
COPY scripts ./scripts
ENTRYPOINT ["gbdf"]
CMD ["--help"]
