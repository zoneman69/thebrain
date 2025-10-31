FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .[dev]

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/train_cloud.py", "--config", "hippo.yaml", "--log-dir", "logs", "--output-dir", "artifacts"]
