FROM python:3.11-slim

WORKDIR /workspace

# System dependencies (IMPORTANT)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-light.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-light.txt

COPY requirements-heavy.txt .
RUN pip install -r requirements-heavy.txt

CMD ["bash"]