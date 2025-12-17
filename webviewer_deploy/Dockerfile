FROM python:3.10-slim

# Install system dependencies for VTK
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path to include src
ENV PYTHONPATH=/app/src

# Run the webviewer
CMD ["python", "-m", "fourc_webviewer.main"]
