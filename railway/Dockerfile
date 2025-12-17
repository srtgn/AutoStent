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

# Copy 4C shared library
COPY --from=fourc /home/user/4C/build/lib4C.so /usr/local/lib/

# Copy MPI libraries from 4C image
COPY --from=fourc /usr/lib/x86_64-linux-gnu/libmpi* /usr/lib/x86_64-linux-gnu/

# Update library cache and set LD_LIBRARY_PATH
RUN ldconfig /usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    numpy \
    stable-baselines3 \
    gymnasium \
    pyvista

# Copy autostent package (for 4C interface)
COPY autostent/ ./autostent/
RUN pip install -e ./autostent

# Copy application code
COPY railway/api_server.py .
COPY railway/mesh_generator.py .
COPY railway/vtu_parser.py .

# Expose port
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run with uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
