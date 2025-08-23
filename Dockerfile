# ---- Builder Stage ----
FROM python:3.13.3-slim-bookworm AS builder

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements and install into the virtual environment
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM python:3.13.3-slim-bookworm

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
ENV VENV_PATH=/opt/venv
COPY --from=builder $VENV_PATH $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy the application code
COPY . .

# Create the outputs directory
RUN mkdir -p outputs

# **NEW: Change ownership of the app directory to the non-root user**
# User 1000, Group 1000 is what's set in job.yaml's securityContext
RUN chown -R 1000:1000 /app

# **NEW: Ensure scripts are executable**
RUN chmod +x /app/main.py

# Command to run the application IN THE CONTAINER
CMD ["python", "main.py", "--config", "/config/molecule_config.yaml"]
