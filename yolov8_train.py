from ultralytics import YOLO

def train_model():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Nano model; change to 'yolov8s.pt', etc., if desired

    # Define the path to your .yaml file
    data_path = 'PATH OF THE data.yaml'

    # Train the model
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='ocean_plastics',
        patience=10,
        device=0  # GPU
    )

    # Evaluate the model on the validation set
    results = model.val()
    print(results)

if __name__ == '__main__':
    train_model()