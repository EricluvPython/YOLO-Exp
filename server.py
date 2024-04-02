import cv2
import numpy as np
import socket
import pickle
import time
import struct
import psutil
import subprocess
import resource
import os

# parameters I can experiment with
cpu_limit_ghz = 1.8
memory_limit_gb = 4
limit_cpu_memory = True

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
print("Loading YOLO")
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in list(net.getUnconnectedOutLayers())]
print("YOLO loaded")

# Socket setup
HOST = '127.0.0.1'  # Server IP
PORT = 5000  # Port to listen on
for conn in psutil.net_connections(kind='all'):
    if str(PORT) in conn.laddr:
        print(f"Closing port {PORT} by terminating PID {conn.pid}")
        process = psutil.Process(conn.pid)
        process.terminate()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Server started, waiting for connection...")

conn, addr = server_socket.accept()
print('Connected by', addr)
total_server_time = 0
while True:
    try:
        # Receive data from client
        print("Receiving data")
        # Receive frame size
        data_size = conn.recv(4)
        if not data_size:
            break
        data_size = struct.unpack('!I', data_size)[0]

        # Receive frame data
        data = b""
        while len(data) < data_size:
            chunk = conn.recv(min(data_size - len(data), 4096))
            if not chunk:
                break
            data += chunk
        print(f"Received data of size {len(data)}")
        frame = pickle.loads(data)

        height, width, channels = frame.shape
        server_time_start = time.time()
        # Detecting objects
        print("Detecting objects")
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process detection results
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
        total_server_time += time.time() - server_time_start
        # Send processed results back to client
        print("Sending back results")
        result_data = pickle.dumps((boxes, confidences, class_ids, classes, colors))
        result_data_size = struct.pack('!I', len(result_data))

        # Send processed results size and data
        conn.sendall(result_data_size)
        conn.sendall(result_data)
        print("\n")
    except Exception as e:
        print("Error:", e)
        break

print(f"Total Server Time: {total_server_time} seconds")
conn.close()
server_socket.close()
