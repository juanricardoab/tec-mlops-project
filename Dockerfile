# Use a base Python image
FROM python:3.10-slim

# Install system dependencies needed for building certain Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into /app in the container
COPY . /app

# Add src directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the default command to run main.py
CMD ["python", "/app/tec_mlops_project/main.py", "--config=params.yaml"]