FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    yt-dlp \
    ffmpeg \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create temp directory
RUN mkdir -p /tmp

# Expose port
EXPOSE 3000

# Run the application
CMD ["python", "app.py"] 