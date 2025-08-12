FROM python:3.11-slim

WORKDIR /app

# Install git for pip packages that require it
RUN apt-get update && apt-get install -y git && \
    apt-get purge -y --auto-remove && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

RUN pip uninstall -y cupy || true \
 && pip install --no-cache-dir "cupy-cuda12x==13.3.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
COPY api/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]