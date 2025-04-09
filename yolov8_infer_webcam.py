from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained YOLOv8 model
model = YOLO(r"PATH OF THE best.pt")


# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define a constant color (Green) for all bounding boxes
BOX_COLOR = (0, 255, 0)  # Green in BGR format

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference on the frame
    results = model(frame, conf=0.5, iou=0.5, verbose=False)

    # Extract results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Get class name

            # Draw bounding box with a fixed green color
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Display label text
            label = f"{class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)

    # Resize for display
    display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

    # Show the output frame
    cv2.imshow('YOLOv8 Live Detection', display_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
