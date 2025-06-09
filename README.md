# iMedSTAM  
Interactive Segmentation and Tracking Anything in 3D Medical Images and Videos.

This repository was created for the [CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/) challenge. It is based on [EfficientTAM](https://github.com/yformer/EfficientTAM).

## Setup

### Installation
```bash
pip install -r requirements.txt
```

### Download Checkpoints
Download the fine-tuned [checkpoint](https://drive.google.com/drive/folders/1z7G0qloQeHoWwT2wJaG3_UibBHn6r4Ey?usp=sharing), or use the original EfficientTAM [checkpoint](https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt). Place the downloaded file in the `checkpoints` directory.

## Training

- Set the path to your training dataset in `configs/efficienttam_training/finetune_all_data.yaml`:
```yaml
dataset:
    # Paths to dataset
    img_folder: null  # Replace with path to training set folder
```

- To fine-tune the base model on the entire dataset using 8 GPUs, run:
```bash
python train.py \
    -c configs/efficienttam_training/finetune_all_data.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

## Inference

You can directly download the [Docker file](https://drive.google.com/drive/folders/1M2t5ny5TLvIPpjXrhK6YQbPQyeqxPHVC?usp=drive_link) prepared for the challenge.

We provide a Dockerfile compatible with the challenge format. For more details, refer to the [challenge website](https://www.codabench.org/competitions/5263/).

To build and save the Docker image:
```bash
docker build -t imedstam:latest .
docker save -o imedstam.tar.gz imedstam:latest
```

You can also run predictions directly using `predict.sh`. Make sure to download the fine-tuned checkpoint and modify the script to include the correct model path: `--model=/your_downloaded_checkpoint`
