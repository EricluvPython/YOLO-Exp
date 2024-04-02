import cv2
import numpy as np
import time
import subprocess
import os
import resource

# parameters I can experiment with
cpu_limit_ghz = 2.8
memory_limit_gb = 8
limit_cpu_memory = True
timeout = 100

# Set CPU limit
if limit_cpu_memory:
    myGHz = 4
    if cpu_limit_ghz >= myGHz:
        cpu_limit = 100
    else:
        cpu_limit = int(cpu_limit_ghz/myGHz*100)
    subprocess.Popen(["cpulimit", "-p", str(os.getpid()), "-l", str(cpu_limit), "-c", "1"])
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_gb * (1024 ** 3), memory_limit_gb * (1024 ** 3)))


# Load YOLO
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in list(net.getUnconnectedOutLayers())]


starttime = time.time()
frametimes = []
# Capture video from camera
cap = cv2.VideoCapture(0)

prev_time = 0
while time.time() - starttime <= 100:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    outs = net.forward(output_layers)
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time
    frametimes.append(inference_time)

    # Showing informations on the screen
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
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    # Calculate framerate
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Display framerate and inference time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), font, 2, (0, 255, 0), 2)
    cv2.putText(frame, f'Inference Time: {round(inference_time, 3)}s', (20, 100), font, 2, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print(f"Average time per frame: {sum(frametimes)/len(frametimes)} seconds of {len(frametimes)} frames")