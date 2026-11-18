# Drishti, CPU image. Build:  docker build -t drishti .
# Run:   docker run -p 8000:8000 drishti   then open http://localhost:8000
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch keeps the image ~3 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# weights + demo videos, then build the search index into the image
RUN python scripts/setup_demo.py && python -m drishti.ingest demo/cameras.json

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
