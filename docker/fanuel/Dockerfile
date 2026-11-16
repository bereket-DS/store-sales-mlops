FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Run validation by default
CMD ["python", "-m", "src.fanuel.train"]