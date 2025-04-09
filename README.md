# 🧠 YOLOv8 Custom Object Detection

This project demonstrates how to train and deploy a custom object detection model using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). It includes training on a custom dataset and real-time inference via webcam.

## 📁 Project Structure

- `yolov8_train.py`: Script to train YOLOv8 on a custom dataset.
- `yolov8_infer_webcam.py`: Script for real-time webcam-based inference.
- `data/data.yaml`: Dataset configuration file.
- `weights/best.pt`: Trained model weights (optional to upload; you can host it externally if needed).

## 🧪 Training the Model

To train the model, run:

```bash
python yolov8_train.py
