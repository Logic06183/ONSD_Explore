FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Copy requirements file
COPY requirements.txt /tmp/pip-tmp/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp

# Install additional OS packages if needed
RUN apt-get update && apt-get install -y \
    build-essential

# [Optional] Uncomment this line to install additional packages
# RUN pip install <additional-package>
