# WORKSHOP-2-Object-detection-using-web-camera
## Aim:

To implement a real-time object detection system using a webcam and YOLOv4 to detect and classify objects for 10 seconds.
## Procedure:

1. Load the pre-trained YOLOv4 network and configuration files.
2. Load the COCO class labels (objects YOLO can detect).
3. Initialize video capture for the webcam and check if it's working.
4. Capture frames from the webcam in a loop.
5. Preprocess each frame and pass it through the YOLOv4 network.
6. Extract object detection results, apply non-max suppression to remove redundant detections.
7. Draw bounding boxes with labels and confidences on detected objects.
8. Stop detection after 10 seconds or if 'q' is pressed.
9. Release the webcam and close display windows.

## Program:
```python
import cv2
import numpy as np
import time

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Try to open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows; for Linux, try cv2.CAP_V4L2 or cv2.CAP_ANY

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Start a timer
start_time = time.time()

while True:
    # Check elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > 50:  # Stop after 10 seconds
        print("Time's up! Stopping object detection.")
        break

    ret, frame = cap.read()

    # Check if frame is captured correctly
    if not ret or frame is None:
        print("Error: Failed to capture frame from webcam.")
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output
    outputs = net.forward(output_layers)

    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
```

## Output:

![image](https://github.com/user-attachments/assets/a10595bf-01df-44f6-8ba3-0ae3b3f6a2be)

## Result:
Thus the object detection system using a webcam has been successfully executed.




