import cv2

#v4l/by-path/platform-rkisp-vir0-video-index0
# Create a VideoCapture object for the camera
#cap = cv2.VideoCapture(0)

gst_str = '/dev/video0'#! video/x-raw'#, width=640, height=480 ! videoconvert ! appsink'
gst_str = 'camera-id=0'
cap = cv2.VideoCapture(0) #cv2.CAP_V4L2)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap = cv2.VideoCapture(1)



# Check if camera is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Capture a frame from the camera
ret, frame = cap.read()

# Release the VideoCapture object
cap.release()

# Display the captured frame on the screen
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

