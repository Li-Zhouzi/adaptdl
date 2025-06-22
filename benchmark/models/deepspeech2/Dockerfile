FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /root

# Install dependencies
COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY benchmark/models/deepspeech2/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install audio processing tools needed for VoxForge
RUN apt-get update && \
    apt-get install -y sox libsox-dev libsox-fmt-all wget bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up your Python paths
COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/deepspeech2 deepspeech2
ENV PYTHONPATH=/root/deepspeech2:$PYTHONPATH

# Working directory for training code
WORKDIR /root/deepspeech2

# Run VoxForge data prep (already limited to 3 samples in your script)
RUN mkdir -p /root/deepspeech2/voxforge_data && \
    python data/voxforge.py --target-dir /root/deepspeech2/voxforge_data
# After running prepare_voxforge.py
RUN python -c "\
import random; \
lines = open('/root/deepspeech2/voxforge_train_manifest.csv').readlines(); \
random.shuffle(lines); \
split = int(0.9 * len(lines)); \
open('/root/deepspeech2/voxforge_train_manifest.csv', 'w').writelines(lines[:split]); \
open('/root/deepspeech2/voxforge_val_manifest.csv', 'w').writelines(lines[split:])"

ENV PYTHONUNBUFFERED=true