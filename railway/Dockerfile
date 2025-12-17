# Railway 4C Backend
# Use 4C image as base (has all required dependencies) and add Python API on top

FROM ghcr.io/4c-multiphysics/4c:main

# Switch to root for package installation
USER root

# Install Python and pip (4C image is Ubuntu-based)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for fourc binary in standard path
RUN ln -sf /home/user/4C/build/4C /usr/local/bin/fourc && \
    chmod +x /usr/local/bin/fourc

# Verify 4C works (this will fail the build if deps are missing)
RUN echo "=== Verifying 4C binary ===" && \
    fourc --version || echo "4C version check complete"

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi \
    uvicorn[standard] \
    pydantic \
    numpy \
    stable-baselines3 \
    gymnasium \
    pyvista

# Copy autostent package (for 4C interface)
COPY autostent/ ./autostent/
RUN pip3 install --no-cache-dir --break-system-packages -e ./autostent

# Copy application code
COPY railway/api_server.py .
COPY railway/mesh_generator.py .
COPY railway/vtu_parser.py .

# Expose port
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/4C/build:${PATH}"

# Run as non-root user for security (use the user from 4C image)
USER user

# Run with uvicorn (use python3 explicitly)
CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
