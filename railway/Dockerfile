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

# Find where 4C libraries are located
RUN echo "=== Finding 4C libraries ===" && \
    find /home/user/4C -name "*.so*" -type f 2>/dev/null | head -5 || echo "Searching..." && \
    ls -la /home/user/4C/build/*.so 2>/dev/null || echo "No .so files in build/" && \
    find /usr -name "*4C*.so*" 2>/dev/null | head -5 || echo "No 4C libs in /usr"

# Verify 4C works (this will fail the build if deps are missing)
RUN echo "=== Verifying 4C binary ===" && \
    export LD_LIBRARY_PATH="/home/user/4C/build:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" && \
    ldconfig /home/user/4C/build /usr/local/lib 2>/dev/null || true && \
    fourc --version || echo "4C version check (may fail without full LD_LIBRARY_PATH)"

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

# Environment variables - CRITICAL: Set LD_LIBRARY_PATH for 4C libraries
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/4C/build:${PATH}"
# Include common library paths where 4C dependencies might be
# Note: Set LD_LIBRARY_PATH directly (don't reference itself)
ENV LD_LIBRARY_PATH="/home/user/4C/build:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib"

# Run as non-root user for security (use the user from 4C image)
USER user

# Run with uvicorn (use PORT env var from Railway)
CMD python3 -m uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8080}
