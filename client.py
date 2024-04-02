import cv2
import socket
import pickle
import time
import struct
import subprocess

# Function to set up network conditions using tcconfig
def setup_network_conditions(delay_ms, loss_percentage, bandwidth_mbps, setlimit, interface="lo"):
    try:
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", interface, "root"])
    except:
        pass
    if setlimit:
        # Set up delay and bandwidth
        subprocess.run(["sudo", "tc", "qdisc", "add", "dev", interface, "root", "netem", "delay", f"{delay_ms}ms", "rate", f"{bandwidth_mbps}mbit", "loss", f"{loss_percentage}%"])

# Simulate network conditions
delay_ms = 30  # Milliseconds
loss_percentage = 1  # Percentage
bandwidth_mbps = 250 # Megabits per second
limit_network = True
setup_network_conditions(delay_ms, loss_percentage, bandwidth_mbps, limit_network)

# Socket setup
HOST = '127.0.0.1'  # Server IP
PORT = 5000  # Port to connect to
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

exp_timeout = 100

try:
    client_socket.connect((HOST, PORT))
    print(f"Connected to server {HOST, PORT}")

    # Capture video from camera
    cap = cv2.VideoCapture(0)

    prev_frame_time = time.time()
    total_start_time = time.time()
    communication_time = 0
    frame_delays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time()-total_start_time > exp_timeout:
            break
        try:
            current_frame_time = time.time()

            # Send frame size and frame data to server
            data = pickle.dumps(frame)
            data_size = struct.pack('!I', len(data))
            client_socket.sendall(data_size)
            client_socket.sendall(data)
            print(f"Sent data of size {len(data)}")

            # Receive processed results from server
            result_data_size = client_socket.recv(4)
            result_data_size = struct.unpack('!I', result_data_size)[0]
            result_data = b""
            while len(result_data) < result_data_size:
                chunk = client_socket.recv(min(result_data_size - len(result_data), 4096))
                if not chunk:
                    break
                result_data += chunk
            print(f"Received data of size {len(result_data)}")
            boxes, confidences, class_ids, classes, colors = pickle.loads(result_data)

            # Visualize detection results
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            time_lag = current_frame_time - prev_frame_time
            prev_frame_time = current_frame_time
            frame_delays.append(time_lag)
            cv2.putText(frame, f'Time Lag: {round(time_lag, 3)}s', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
            print("\n")
        except Exception as e:
            print("Error:", e)
            break

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - total_start_time

    print(f"Total Time: {total_time} seconds")
    print(f"Average frame rate: {sum(frame_delays)/len(frame_delays)} of {len(frame_delays)} frames")


except Exception as e:
    print("Error:", e)

finally:
    client_socket.close()
