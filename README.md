# 🧠 Plastic Waste Detection Using YOLOv8

This project uses Ultralytics YOLOv8 to train a model for detecting plastic waste and deploys it in real-time using a webcam.

## 📁 Project Structure

- `yolov8_train.py`: Train YOLOv8 on custom data
- `yolov8_infer_webcam.py`: Run real-time detection via webcam
- `data/data.yaml`: Dataset configuration
- `weights/best.pt`: Trained model (optional - link if hosted externally)

## 🧪 Training

```bash
python yolov8_train.py
