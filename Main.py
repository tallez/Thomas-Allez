# Hello ! Welcome to the Main Vision Script.
# This script aims to locate waste flowing down a river from a video feed. 
# With a socket module, it has to transmit x,y + speed through internet

# A.1 Import available modules
import cv2
import numpy as np
from tracker import EuclideanDistTracker


# A.2 Import side scripts


# A.3 Create and connect Socket for transmission

# B. Get video feed

Video_Feed = cv2.VideoCapture(0) # Put cv2.VideoCapture(1) to get livefeed from main camera. 

# C. Get video area of interest 

# C.1 Delimitation of the Area

# Only a portion of the camera field of view is interesting for analysis, it has to be very still and offer a stable background for contrasting with targets
# Delimitations of the Area of interest has to be define precisely, a ruler or markers should be considered to appear on the video feed for calibration. 

# For test we will agree to the following values
# Positions in px of the corners on the video
# lt : x:252, y:800 (LeftTop)
# rt : x:1653, y:802 (RightTop)
# rb : x:1824, y:1073 (RightBottom)
# lb : x:107, y:1066 (LeftBottom)
# Represent a rectangle on the field (10x70cm)

lt = (252,800)
rt = (1653,802)
rb = (1824,1073)
lb = (107,1066)

Area_0f_Interest = [lt,rt,rb,lb]

# C.2 Array Markers of the Area
# To warp the video so the transmitted positions are given according to an orthonormal coordinates system. 

Affine_Markers = np.array(eval(str(Area_0f_Interest)), dtype = "float32")

# D Object Detection Methods
# D.1 Creation of the Markers list and position registration

Trackers = EuclideanDistTracker()

# D.2 Filter video for movement detection

Filter_Method = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# E. Video Analysis
# E.1 Monitor frame count + create dataframe for flow speed analysis + Create a finalized dataformat to send

frame_count = 0
flow_frame = []
Data_Pack = []

# E.2 Core Analysis

while True:
    ret, frame = Video_Feed.read()
    frame_count = frame_count + 1
    Detections = []

    # Affine Transform of the Area of Interest
    
   

    # Movement filtering + Contours finding

    Mask = Filter_Method.apply(frame)
    _, Mask = cv2.threshold(Mask, 100, 200, cv2.THRESH_BINARY)
    Contours, _ = cv2.findContours(Mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in Contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            Detections.append([x, y, w, h])
    
    # 2. Object Tracking
    boxes_ids = Trackers.update(Detections)
    for box_id in boxes_ids:

        x, y, w, h, id = box_id # Box_ID gives detected objet size and position

        
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

Video_Feed.release()
cv2.destroyAllWindows()


