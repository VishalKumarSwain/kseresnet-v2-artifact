FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy grpcio grpcio-tools
COPY main.py .
COPY crash_model_v4.pth .
COPY competition_2026_pb2.py .
COPY competition_2026_pb2_grpc.py .
EXPOSE 50051
ENTRYPOINT ["python", "main.py"]
