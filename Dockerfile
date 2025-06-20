# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p results/pipeline_$(date +%Y%m%d_%H%M%S) \
    results/data_quality \
    results/visualizations \
    dataset/explore/results/data_quality

# Set permissions
RUN chmod +x run_pipeline.py \
    && chmod +x ag_news_pipeline.py \
    && chmod +x run_both_pipelines.py

# Expose port (if needed for any web interface)
EXPOSE 8000

# Set the default command to run both pipelines
CMD ["python", "run_both_pipelines.py"]
