# Player Re-Identification in Soccer Matches

## Overview
This project addresses the problem of player re-identification in soccer match videos. The objective is to assign consistent IDs to each player, even if they leave and reenter the frame at different points during the match.

## Setup Instructions

### Dependencies
Install the following Python packages:
```bash
pip install ultralytics opencv-python torch torchvision numpy scipy
```
## Directory Structure
```bash

project_root/
├── data/                    # Input videos
├── models/
│   └── yolo/
│       └── best.pt          # Provided YOLOv11 model (fine-tuned)
├── output/                  # Annotated output videos
├── src/
│   ├── detector.py          # Loads YOLOv11 and returns bounding boxes
│   ├── tracker.py           # Assigns and updates player IDs
│   ├── feature_extractor.py# Extracts embeddings using ResNet-18
│   └── reidentifier.py      # Matches embeddings to maintain ID consistency
├── main.py                  # Runs the full pipeline
└── README.md
```


## How to Run

Follow these steps to execute the player re-identification pipeline:

1. Ensure the YOLOv11 model weights (`best.pt`) are placed in the correct path

2. Place your input video inside the `data/` directory:

3. Run the following command from the root directory of the project:
```bash
python main.py --video data/sample.mp4 --output output/result.mp4 --yolo-weights models/yolo/best.pt

```
## Notes

- The detection model used is a fine-tuned YOLOv11, trained specifically to detect **players** and **ball** in soccer videos.
- Player tracking is initially handled using a **centroid-based tracker**, which assigns temporary IDs frame by frame.
- For consistent identity assignment over time, the system employs **appearance-based re-identification**:
  - Cropped player images are passed through a **ResNet-18** model to extract embeddings.
  - **Cosine similarity** is used to match current embeddings with a memory bank of past embeddings.

