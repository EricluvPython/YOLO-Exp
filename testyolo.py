import cv2
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import resource
import os

# Function to set CPU and memory quotas
def set_resource_limits(cpu_percent, memory_percent):
    # Set CPU limit
    psutil.Process(os.getpid()).cpu_percent(cpu_percent)

    # Set memory limit
    mem_limit = int(psutil.virtual_memory().total * (memory_percent / 100))
    soft, hard = resource.RLIM_INFINITY, resource.RLIM_INFINITY
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, hard))

# Function to measure YOLO inference time
def measure_inference_time(net, image):
    # Forward pass
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layer_outputs = net.forward(get_output_layers(net))
    end_time = time.time()
    return end_time - start_time

# Function to get output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i-1] for i in list(net.getUnconnectedOutLayers())]

# Function to load YOLO model
def load_yolo_model():
    # Load YOLO weights and configuration
    net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
    print("Loaded model")
    return net

# Function to test YOLO performance under different conditions
def test_yolo_performance(video_path, scene_complexity, resolution, frame_rate,  cpu_percent, memory_percent, timeout=10):
    print(f"Testing with {scene_complexity} {resolution} {frame_rate} {cpu_percent} {memory_percent}")
    
    # Set CPU and memory quotas
    set_resource_limits(cpu_percent, memory_percent)

    # Load YOLO model
    net = load_yolo_model()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Set resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    # Initialize lists to store inference times
    inference_times = []
    
    timer = time.time()
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret or time.time()-timer > timeout:
            break

        # Measure inference time
        inference_time = measure_inference_time(net, frame)
        inference_times.append(inference_time)
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)

    # Close video file
    cap.release()
    print(f"Got {len(inference_times)} frames in {time.time()-timer} seconds")
    return avg_inference_time

def run_tests():
    # Define test parameters
    video_paths = ["low_complexity_video.mp4", "medium_complexity_video.mp4", "high_complexity_video.mp4"]
    scene_complexities = ["Low", "Medium", "High"]
    resolutions = [(640, 360), (1280, 720)]
    frame_rates = [30, 60]
    cpu_limits = [25, 50, 100]
    memory_limits = [25, 50, 100]

    results = []

    # Run tests
    for video_path, scene_complexity in zip(video_paths, scene_complexities):
        for resolution in resolutions:
            for frame_rate in frame_rates:
                for cpu_limit in cpu_limits:
                    for memory_limit in memory_limits:
                        inference_time = test_yolo_performance(video_path, scene_complexity, resolution, frame_rate, cpu_limit, memory_limit)
                        results.append([video_path, scene_complexity, resolution[0], resolution[1], frame_rate, cpu_limit, memory_limit, inference_time])

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Video", "Scene Complexity", "Resolution Width", "Resolution Height", "Frame Rate", "CPU Limit (%)", "Memory Limit (%)", "Inference Time"])
    df.to_csv("yolo_performance_results.csv", index=False)

    return df

# Main function
if __name__ == "__main__":
    results_df = run_tests()
    print("Results saved to yolo_performance_results.csv")
