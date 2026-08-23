FROM python:3.11-slim
# Tesseract is the OCR backend (ADR-0001: no Textract).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-ita \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY graphify ./graphify
COPY graphify_ent ./graphify_ent
RUN pip install --no-cache-dir -e ".[ent,pdf,neo4j,office]"
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "graphify_ent.server.app"]
