FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Create directories for outputs
RUN mkdir -p outputs/checkpoints outputs/logs outputs/samples

CMD ["/bin/bash"]
