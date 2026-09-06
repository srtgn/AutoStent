FROM python:3.10-slim

# Install system dependencies for VTK offscreen rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libxrender1 \
    libglib2.0-0 \
    libosmesa6 \
    libegl1 \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path to include src
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Run the webviewer
CMD ["./start.sh"]
