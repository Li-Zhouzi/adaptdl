FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/bert/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/bert bert
RUN curl -o /root/bert/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json && \
    curl -o /root/bert/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
ENV PYTHONPATH=/root/bert:$PYTHONPATH
WORKDIR /root/bert

ENV PYTHONUNBUFFERED=true
