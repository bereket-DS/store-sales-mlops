FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/tehreem-requirements.txt
RUN pip install --no-cache-dir -r /tmp/tehreem-requirements.txt
WORKDIR /app
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m src.tehreem.prepare && python -m src.tehreem.train"]
