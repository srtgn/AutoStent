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

# Copy 4C binary and libraries
COPY --from=fourc /home/user/4C/build/4C /usr/local/bin/fourc
COPY --from=fourc /usr/lib/x86_64-linux-gnu/libmpi* /usr/lib/x86_64-linux-gnu/
COPY --from=fourc /usr/lib/x86_64-linux-gnu/openmpi /usr/lib/x86_64-linux-gnu/openmpi

# Make executable
RUN chmod +x /usr/local/bin/fourc

WORKDIR /app

# Install Python dependencies
COPY railway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API server
COPY railway/api_server.py .
COPY autostent/ ./autostent/

# Environment
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "api_server.py"]
