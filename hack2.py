import cv2
import numpy as np
import time
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\prash\OneDrive\Desktop\hack.h5")

# Load YOLO
net = cv2.dnn.readNet(r"C:\Users\prash\OneDrive\Desktop\yolov3.weights", r"C:\Users\prash\OneDrive\Desktop\yolov3 (1).cfg")
classes = []
with open(r"C:\Users\prash\OneDrive\Desktop\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Flatten the array if it's not already flattened
if unconnected_out_layers.ndim == 2:
    unconnected_out_layers = unconnected_out_layers.flatten()

output_layers = [layer_names[i - 1] for i in unconnected_out_layers] if len(unconnected_out_layers) > 0 else []

cap = cv2.VideoCapture(0)

# Variables for tracking drones
drones_detected = {}

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

            # Extract drone image and perform detection
            drone_image = frame[y:y+h, x:x+w]
            drone_image_resized = cv2.resize(drone_image, (256, 256))  # Resize the image to match model's input shape
            detection_result = model.predict(np.expand_dims(drone_image_resized, axis=0))

            # Check if the prediction is greater than 0.5
            if detection_result[0][0] > 0.5:
                label = "drone"

            # If a drone is detected, track it and perform actions only if result is positive
            if label == "drone":
                drone_id = f"{x}_{y}_{w}_{h}"
                if drone_id not in drones_detected:
                    # Track the drone
                    drones_detected[drone_id] = {"start_time": time.time(), "end_time": None}
                    print("Drone detected! Taking action...")
                else:
                    drones_detected[drone_id]["end_time"] = time.time()
                    duration = drones_detected[drone_id]["end_time"] - drones_detected[drone_id]["start_time"]
                    print(f"Drone detected for {duration} seconds")
                    del drones_detected[drone_id]

    # Display the result
    cv2.imshow("YOLO Object Detection", frame)

    # Wait for key press
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
