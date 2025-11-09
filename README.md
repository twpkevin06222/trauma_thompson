# The Trauma THOMPSON Challenge 2025
Trauma TeleHelper for Operational Medical Procedure Support and Offline Network <br>
for more info please refer to the official website of the grand challenge: [link](https://t3challenge25.grand-challenge.org/t3challenge25/)
## Objective
The Trauma THOMPSON Challenge 2025 aims to drive innovation in AI-assisted remote instruction systems that can provide real-time guidance to frontline providers during high-stakes and lifesaving procedures. The dataset is available for download with the consent of authors [here](https://t3challenge25.grand-challenge.org/data/). I by all means have no rights nor am I authorised to distribute the data or expose the dataset.

## Tasks
Full description of the task is available from the official [website](https://t3challenge25.grand-challenge.org/task/).
- Task 1: Action recognition and anticipation of regular and just in-time procedures
- Task 2: Emergency procedure hand tracking
- Task 3: Emergency procedure tool detection
- Task 4: Realism assessment
- Task 5: Visual Question Answering

## Setup
```
uv init trauma_thompson --python 3.12
uv pip install requirements.txt   
```

## Hardware Specs
| Component | Specification |
|------------|----------------|
| CPU | Intel Core i7-14700K |
| GPU | NVIDIA RTX 5090|
| RAM | 64 GB |
| OS | Ubuntu 24.04 LTS |
| CUDA | 12.8 |

# Results
## Task 2 - Hand Tracking
Utilised [Ultralytics](https://docs.ultralytics.com/modes/track/) YOLOv11n for object detection training and infer using Ultralytics's object tracking method. The provided dataset consisted of annotations that are not suitable for training (refer to task2 subfolder EDA section for more info) and was subsequently removed from the training and validation set. <br> 
For some results, the training results for single class object detection is not great.
<p align="center">
<img src="assets/task2_metrics.png" width=800, height=400><br>
</p>
<p align="center">
<img src="assets/task2_example.gif" width=800, height=500>
</p>


## Task 3 - Tools Detection
Utilised Ultralytics framework YOLOv11n for quick prototyping of object detection training. Dataset spllits can refer to task3 subfolder EDA section.
<p align="center">
<img src="assets/task3_metrics.png" width=800, height=400><br>
</p>
<p align="center">
<img src="assets/task3_example.gif" width=800, height=500>
</p>
