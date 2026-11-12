import cv2

# Pipeline yang benar untuk OpenCV
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test.jpg", frame)
        print("✅ Image saved: test.jpg")
    cap.release()
else:
    print("❌ Cannot open camera")