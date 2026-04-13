FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY scripts/ scripts/

# Training data and experiment configs are mounted at runtime:
#   -v /path/to/data:/app/data
#   -v /path/to/experiments:/app/data/experiments

ENTRYPOINT ["python"]
CMD ["scripts/04_train.py"]
