import subprocess
import numpy as np
import cv2

command = [
    "gst-launch-1.0", "-q",
    "nvarguscamerasrc", "sensor-id=0", "!",
    "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1", "!",
    "nvvidconv", "!",
    "video/x-raw,width=1280,height=720,format=(string)BGRx", "!",
    "videoconvert", "!",
    "video/x-raw,format=(string)BGR", "!",
    "fdsink", "fd=1"
]

print("Starting GStreamer subprocess...")
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Reading frame...")
raw_image = proc.stdout.read(1280 * 720 * 3)
err = proc.stderr.read().decode('utf-8')
print("STDERR:")
print(err)

if len(raw_image) == 1280 * 720 * 3:
    image = np.frombuffer(raw_image, dtype=np.uint8).reshape((720, 1280, 3))
    print(f"SUCCESS! Got frame with shape: {image.shape}")
else:
    print(f"FAILED! Read {len(raw_image)} bytes")

proc.terminate()

