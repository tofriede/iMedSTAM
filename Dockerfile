# use an appropriate base image with GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /workspace

# install python dependencies
COPY requirements_infer.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY infer.py /workspace/infer.py
COPY template/inference.py /workspace/template/inference.py
COPY template/get_boxes.py /workspace/template/get_boxes.py
COPY template/error.py /workspace/template/error.py
COPY efficient_track_anything /workspace/efficient_track_anything
COPY checkpoints/coreset_checkpoint.pt /workspace/model_final.pth

COPY predict.sh /workspace/predict.sh
RUN chmod +x /workspace/predict.sh

# set default command
CMD ["/bin/bash"]