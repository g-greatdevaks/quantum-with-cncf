# ---- Final Stage ----
# Purpose: Run the application with CUDA support
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

WORKDIR /app

# Set non-interactive frontend for apt commands
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gnupg \
    ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
        build-essential \
        libgomp1 \
    && apt-get remove -y software-properties-common wget gnupg \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/deadsnakes*

# Make python3.13 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install pip for Python 3.13 using ensurepip
RUN python3 -m ensurepip --upgrade

# Upgrade pip and install setuptools and wheel
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements and install into the venv
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Change ownership and permissions
RUN chown -R 1000:1000 /app && chown -R 1000:1000 /opt/venv
RUN chmod +x /app/main.py

# Switch to the non-root user
USER 1000

# Execute using the venv python
CMD ["/opt/venv/bin/python", "main.py", "--config", "/config/molecule_config.yaml"]
