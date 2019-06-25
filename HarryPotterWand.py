import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2

# initializing Picamera
camera = PiCamera()
camera.framerate = 33
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size = (640, 480))


# Define parameters for the required blob
params = cv2.SimpleBlobDetector_Params()

# setting the thresholds
params.minThreshold = 150
params.maxThreshold = 250

# filter by color
params.filterByColor = 1
params.blobColor = 255

# filter by circularity
params.filterByCircularity = 1
params.minCircularity = 0.68

# filter by area
params.filterByArea = 1
params.minArea = 30
# params.maxArea = 1500

# creating object for SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)


flag = 0
points = []
lower_green = np.array([0, 255, 0])
upper_green = np.array([0, 255, 0])


def show_last(img):
    cv2.imwrite("/home/pi/Desktop/lastframe.jpg", img)

time.sleep(0.1)

for image in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    frame = image.array
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #detecting keypoints
    keypoints = detector.detect(frame)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    #starting and ending circle
    frame_with_keypoints = cv2.circle(frame_with_keypoints, (190, 40), 6, (0, 255, 0), 2)
    frame_with_keypoints = cv2.circle(frame_with_keypoints, (190, 160), 6, (0, 0, 255), 2)


    #points_array = cv2.KeyPoint_convert(keypoints)
    points_array = cv2.KeyPoint_convert(keypoints)



    if flag == 1:
        # Get coordinates of the center of blob from keypoints and append them in points list
        points.append(points_array[0])


        # Draw the path by drawing lines between 2 consecutive points in points list
        for i in range(1, len(points)):
            cv2.line(frame_with_keypoints, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), 5)


    if len(points_array) != 0:

        if flag == 1:
            if int(points_array[0][0]) in range(189, 191) and int(points_array[0][1]) in range(159, 160):
                print("khatam ho gaya!!")
                frame_with_keypoints = cv2.inRange(frame_with_keypoints, lower_green, upper_green)
                show_last(frame_with_keypoints)
                break

        if flag == 0:
            if int(points_array[0][0]) in range(189,191) and int(points_array[0][1]) in range(39, 41):
                print("pahuch gaya!!")
                flag = 1    

                
    cv2.imshow("video",frame_with_keypoints)
    cv2.imshow("video 2",frame)
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
