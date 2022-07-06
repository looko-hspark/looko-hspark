FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt-get update
WORKDIR cd
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt