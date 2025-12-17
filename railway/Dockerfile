# Railway 4C Backend
# Multi-stage build: 4C binaries + Python API server

FROM ghcr.io/4c-multiphysics/4c:main AS fourc

# Python API stage
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy 4C binary from official image
COPY --from=fourc /home/user/4C/build/4C /usr/local/bin/fourc
RUN chmod +x /usr/local/bin/fourc

# Copy essential shared libraries from 4C image
COPY --from=fourc /usr/lib/x86_64-linux-gnu/libmpi* /usr/lib/x86_64-linux-gnu/

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    numpy \
    stable-baselines3 \
    gymnasium

# Copy autostent package (for 4C interface)
COPY autostent/ ./autostent/
RUN pip install -e ./autostent

# Copy application code
COPY railway/api_server.py .

# Expose port
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run with uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
