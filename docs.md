# Docs
## 1. System Selection and Setup
To select the most appropiate model for this project I have done a research of the state of art model in object detection tasks. Detecting a single-class object (ultrasonic probe) in drone imagery requires a model that balances accuracy and speed, especially for deployment on IoT boards like NVIDIA Jetson devices.

Key considerations include: high mAP (mean Average Precision) for reliable probe detection, real-time inference speed (frames per second) on Jetson devices and reasonable model size / memory usage. It is important as well that it has a manageable training time. 

State-of-the-art lightweight detections are YOLO family. I decided to take YOLOv8 and not newer version (v9-v12) because of their lower maturity and community adoption. (lower stability, community support, and reproducibility)

Quote: https://www.stereolabs.com/en-fr/blog/performance-of-yolo-v5-v7-and-v8
https://arxiv.org/html/2409.16808v1#S5

## 2. Training and Fine-Tuning
Train or fine-tune the system on the provided dataset.
Document the training process, including any hyperparameters used (e.g., learning rate, batch size, …).

# REMOVE ME
1.	Dataset analysis
2.	Best models for given dataset and hardware.
3.	Dataset preprocessing.
4.	Training and inference on a base-line model.
5.	Training and inference on selected model.
6.	Comparison.
7.	Data augmentation techniques.
8.	Conclusions and comparison.
