#!/usr/bin/env python
# coding: utf-8

# from imutils.video import FPS
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
# 
# webcam = 1
# expected_confidence = 0.3
# threshold = 0.1
# show_output = 1
# save_output = 1
# kernel = np.ones((5,5),np.uint8)
# writer = None
# fps = FPS().start()
# 
# weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
# configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
# 
# print("[INFO] loading Mask R-CNN from disk...")
# net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
# 
# if use_gpu:
#     # set CUDA as the preferable backend and target
#     print("[INFO] setting preferable backend and target to CUDA...")
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# 
# print("[INFO] accessing video stream...")
# cap = cv2.VideoCapture(0)
# 
# print("[INFO] background recording...")
# for _ in range(60):
#     _,bg = cap.read()
# print("[INFO] background recording done...")
# 
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter('output.avi', fourcc, 20,(bg.shape[1], bg.shape[0]), True)
# 
# while True:
#     grabbed, frame = cap.read()
#     cv2.imshow('org',frame)
#     if not grabbed:
#         break
# 
#     blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
#     net.setInput(blob)
#     (boxes, masks) = net.forward(["detection_out_final","detection_masks"])
#     for i in range(0, boxes.shape[2]):
#         classID = int(boxes[0, 0, i, 1])
#         if classID!=0:continue
#         confidence = boxes[0, 0, i, 2]
# 
#         if confidence > expected_confidence:
#             (H, W) = frame.shape[:2]
#             box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
#             (startX, startY, endX, endY) = box.astype("int")
#             boxW = endX - startX
#             boxH = endY - startY
#             mask = masks[i, classID]
#             mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_CUBIC)
#             mask = (mask > threshold)
#             bwmask = np.array(mask,dtype=np.uint8) * 255
#             bwmask = np.reshape(bwmask,mask.shape)
#             bwmask = cv2.dilate(bwmask,kernel,iterations=1)
# 
#             frame[startY:endY, startX:endX][np.where(bwmask==255)] = bg[startY:endY, startX:endX][np.where(bwmask==255)]
# 
#     if show_output:
#         cv2.imshow("Frame", frame)
# 
#         if cv2.waitKey(1) ==27:
#             break
# 
#     if save_output:
#         writer.write(frame)
# 
#     fps.update()
# 
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# In[2]:


pip install imutils


# In[3]:


from imutils.video import FPS
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

webcam = 1
expected_confidence = 0.3
threshold = 0.1
show_output = 1
save_output = 1
kernel = np.ones((5,5),np.uint8)
writer = None
fps = FPS().start()

weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

if use_gpu:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(0)

print("[INFO] background recording...")
for _ in range(60):
    _,bg = cap.read()
print("[INFO] background recording done...")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 20,(bg.shape[1], bg.shape[0]), True)

while True:
    grabbed, frame = cap.read()
    cv2.imshow('org',frame)
    if not grabbed:
        break

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final","detection_masks"])
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        if classID!=0:continue
        confidence = boxes[0, 0, i, 2]

        if confidence > expected_confidence:
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_CUBIC)
            mask = (mask > threshold)
            bwmask = np.array(mask,dtype=np.uint8) * 255
            bwmask = np.reshape(bwmask,mask.shape)
            bwmask = cv2.dilate(bwmask,kernel,iterations=1)

            frame[startY:endY, startX:endX][np.where(bwmask==255)] = bg[startY:endY, startX:endX][np.where(bwmask==255)]

    if show_output:
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) ==27:
            break

    if save_output:
        writer.write(frame)

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# In[5]:


from imutils.video import FPS
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set the webcam index
webcam = 1

# Set parameters
expected_confidence = 0.3
threshold = 0.1
show_output = 1
save_output = 1
kernel = np.ones((5, 5), np.uint8)
writer = None
fps = FPS().start()

# Specify the paths to the Mask R-CNN model files
weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# Print full paths for debugging
print("Weights Path:", os.path.abspath(weightsPath))
print("Config Path:", os.path.abspath(configPath))

# Check if GPU should be used (change use_gpu to True if using GPU)
use_gpu = False

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

if use_gpu:
    # Set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Access the video stream
print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(webcam)

# Record the background frame
print("[INFO] background recording...")
for _ in range(60):
    _, bg = cap.read()
print("[INFO] background recording done...")

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 20, (bg.shape[1], bg.shape[0]), True)

# Main processing loop
while True:
    grabbed, frame = cap.read()
    cv2.imshow('org', frame)
    if not grabbed:
        break

    # Forward pass through the Mask R-CNN network
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # Process each detected instance
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        if classID != 0:
            continue

        confidence = boxes[0, 0, i, 2]

        if confidence > expected_confidence:
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > threshold)
            bwmask = np.array(mask, dtype=np.uint8) * 255
            bwmask = np.reshape(bwmask, mask.shape)
            bwmask = cv2.dilate(bwmask, kernel, iterations=1)

            # Replace the pixels in the frame with the background pixels based on the mask
            frame[startY:endY, startX:endX][np.where(bwmask == 255)] = bg[startY:endY, startX:endX][
                np.where(bwmask == 255)]

    if show_output:
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    if save_output:
        writer.write(frame)

    fps.update()

# Release resources
fps.stop()
cap.release()
cv2.destroyAllWindows()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# In[6]:


pip install --upgrade opencv-python


# In[7]:


from imutils.video import FPS
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set the webcam index
webcam = 1

# Set parameters
expected_confidence = 0.3
threshold = 0.1
show_output = 1
save_output = 1
kernel = np.ones((5, 5), np.uint8)
writer = None
fps = FPS().start()

# Specify the paths to the Mask R-CNN model files
weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# Print full paths for debugging
print("Weights Path:", os.path.abspath(weightsPath))
print("Config Path:", os.path.abspath(configPath))

# Check if GPU should be used (change use_gpu to True if using GPU)
use_gpu = False

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

if use_gpu:
    # Set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Access the video stream
print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(webcam)

# Record the background frame
print("[INFO] background recording...")
for _ in range(60):
    _, bg = cap.read()
print("[INFO] background recording done...")

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 20, (bg.shape[1], bg.shape[0]), True)

# Main processing loop
while True:
    grabbed, frame = cap.read()
    cv2.imshow('org', frame)
    if not grabbed:
        break

    # Forward pass through the Mask R-CNN network
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # Process each detected instance
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        if classID != 0:
            continue

        confidence = boxes[0, 0, i, 2]

        if confidence > expected_confidence:
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > threshold)
            bwmask = np.array(mask, dtype=np.uint8) * 255
            bwmask = np.reshape(bwmask, mask.shape)
            bwmask = cv2.dilate(bwmask, kernel, iterations=1)

            # Replace the pixels in the frame with the background pixels based on the mask
            frame[startY:endY, startX:endX][np.where(bwmask == 255)] = bg[startY:endY, startX:endX][
                np.where(bwmask == 255)]

    if show_output:
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    if save_output:
        writer.write(frame)

    fps.update()

# Release resources
fps.stop()
cap.release()
cv2.destroyAllWindows()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# In[8]:


# Specify the paths to the OpenVINO IR files
irModelPath = "mask-rcnn-coco/frozen_inference_graph.xml"
irWeightsPath = "mask-rcnn-coco/frozen_inference_graph.bin"

# Load the OpenVINO IR model
net = cv2.dnn.readNet(irModelPath, irWeightsPath)


# In[2]:


import cv2
import numpy as np

# Function to count fingers
def count_fingers(hand_contour):
    epsilon = 0.04 * cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour, epsilon, True)

    defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))

    if defects is not None:
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
            
            if angle <= np.pi/2:
                count_defects += 1

        return count_defects
    else:
        return 0

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use thresholding to segment the hand
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 1000:
            # Convex hull
            hull = cv2.convexHull(contour, returnPoints=False)
            
            # Count fingers
            finger_count = count_fingers(contour)
            
            # Draw the hand contour and the number of fingers
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            cv2.putText(frame, str(finger_count + 1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Recognition', frame)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    if hand_contour is not None:
        # Draw the hand contour
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[10]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def count_fingers(hand_contour):
    epsilon = 0.02 * cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour, epsilon, True)

    defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))

    if defects is not None:
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            
            if angle <= np.pi/2:
                finger_count += 1

        return finger_count

    return 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    if hand_contour is not None:
        # Count fingers
        finger_count = count_fingers(hand_contour)

        # Draw the hand contour and the number of fingers
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the sum of the first two fingers
        sum_of_first_two_fingers = min(2, finger_count)
        cv2.putText(frame, f'Sum of first two fingers: {sum_of_first_two_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[16]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def draw_hand_link(frame, hand_contour):
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    # Draw the hand link (rectangle)
    draw_hand_link(frame, hand_contour)

    cv2.imshow('Hand Link Animation', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[21]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def draw_hand_link(frame, hand_contour):
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_blast_effect(frame, hand_contour, blast_radius):
    if hand_contour is not None:
        # Get the center of the bounding box of the hand contour
        center_x = int(cv2.boundingRect(hand_contour)[0] + cv2.boundingRect(hand_contour)[2] / 2)
        center_y = int(cv2.boundingRect(hand_contour)[1] + cv2.boundingRect(hand_contour)[3] / 2)

        # Draw a circle (blast effect) at the center
        cv2.circle(frame, (center_x, center_y), blast_radius, (255, 0, 0), -1)

cap = cv2.VideoCapture(0)
blast_radius = 50  # Adjust the blast radius as needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    # Draw the hand link (rectangle)
    draw_hand_link(frame, hand_contour)

    # Draw the blast effect when the hand is open
    if hand_contour is not None and cv2.contourArea(hand_contour) > 10000:  # Adjust the area threshold as needed
        draw_blast_effect(frame, hand_contour, blast_radius)

    cv2.imshow('Hand Blast Animation', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def draw_explosion(frame, center):
    # Draw a simple explosion effect (circle)
    radius = 50
    cv2.circle(frame, center, radius, (255, 0, 0), -1)

def draw_hand_link(frame, hand_contour):
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

explosion_center = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    if hand_contour is not None:
        # Draw the hand link (rectangle)
        draw_hand_link(frame, hand_contour)

        # Check if palm is open (you can customize this condition)
        if cv2.contourArea(hand_contour) > 5000:
            # Set the explosion center to the center of the bounding box of the hand
            explosion_center = ((x + x + w) // 2, (y + y + h) // 2)

        # Draw explosion if center is set
        if explosion_center is not None:
            draw_explosion(frame, explosion_center)

    cv2.imshow('Hand Fun Animation', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
import random

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def draw_explosion(frame, center):
    # Draw a dynamic explosion effect using circles of different sizes and colors
    for _ in range(10):
        radius = random.randint(10, 50)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(frame, center, radius, color, -1)

def draw_hand_link(frame, hand_contour):
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

explosion_center = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find hand contour
    hand_contour = find_hand(frame)

    if hand_contour is not None:
        # Draw the hand link (rectangle)
        draw_hand_link(frame, hand_contour)

        # Check if palm is open (you can customize this condition)
        if cv2.contourArea(hand_contour) > 5000:
            # Set the explosion center to the center of the bounding box of the hand
            explosion_center = ((x + x + w) // 2, (y + y + h) // 2)

        # Draw explosion if center is set
        if explosion_center is not None:
            draw_explosion(frame, explosion_center)

    cv2.imshow('Hand Fun Animation', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[1]:


import cv2
import pyttsx3
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def gesture_recognition(hand_contour):
    # Check if the hand is open or closed based on contour area
    if cv2.contourArea(hand_contour) > 5000:
        return "open"
    else:
        return "closed"

def main():
    cap = cv2.VideoCapture(0)

    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    hand_status = "closed"
    previous_hand_status = "closed"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Find hand contour
        hand_contour = find_hand(frame)

        if hand_contour is not None:
            # Perform hand gesture recognition
            hand_status = gesture_recognition(hand_contour)

            # Speak greetings based on hand status change
            if hand_status != previous_hand_status:
                if hand_status == "open":
                    engine.say("Hi there!")
                else:
                    engine.say("Goodbye!")

                # Wait for the speech to finish before capturing new gestures
                engine.runAndWait()

            previous_hand_status = hand_status

            # Draw hand contour for visualization
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[2]:


pip install pyttsx3


# In[ ]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def gesture_recognition(hand_contour):
    # Check if the hand is open or closed based on contour area
    if cv2.contourArea(hand_contour) > 5000:
        return "open"
    else:
        return "closed"

def main():
    cap = cv2.VideoCapture(0)

    hand_status = "closed"
    previous_hand_status = "closed"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Find hand contour
        hand_contour = find_hand(frame)

        if hand_contour is not None:
            # Perform hand gesture recognition
            hand_status = gesture_recognition(hand_contour)

            # Print greetings based on hand status change
            if hand_status != previous_hand_status:
                if hand_status == "open":
                    print("Hi there!")
                else:
                    print("Goodbye!")

            previous_hand_status = hand_status

            # Draw hand contour for visualization
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":


# In[5]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def gesture_recognition(hand_contour):
    # Check if the hand is open or closed based on contour area
    if cv2.contourArea(hand_contour) > 5000:
        return "open"
    else:
        return "closed"

def main():
    cap = cv2.VideoCapture(0)

    hand_status = "closed"
    previous_hand_status = "closed"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Find hand contour
        hand_contour = find_hand(frame)

        if hand_contour is not None:
            # Perform hand gesture recognition
            hand_status = gesture_recognition(hand_contour)

            # Print hand status for debugging
            print("Hand Status:", hand_status)

            # Draw hand contour for visualization
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[8]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def count_fingers(hand_contour):
    epsilon = 0.02 * cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour, epsilon, True)

    defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))

    if defects is not None:
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            
            # Check if points exist in the approx array
            if s < len(approx) and e < len(approx) and f < len(approx):
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                
                # Angle threshold to consider as a finger
                if angle <= np.pi/2:
                    finger_count += 1

        return finger_count
    else:
        return 0

def gesture_recognition(hand_contour):
    # Check if the hand is open or closed based on finger count
    fingers = count_fingers(hand_contour)
    if fingers >= 3:
        return "open"
    else:
        return "closed"

def main():
    cap = cv2.VideoCapture(0)

    hand_status = "closed"
    previous_hand_status = "closed"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Find hand contour
        hand_contour = find_hand(frame)

        if hand_contour is not None:
            # Perform hand gesture recognition
            hand_status = gesture_recognition(hand_contour)

            # Print hand status for debugging
            print("Hand Status:", hand_status)

            # Draw hand contour and finger count for visualization
            finger_count = count_fingers(hand_contour)
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[9]:


import cv2
import numpy as np

def find_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

def count_fingers(hand_contour):
    epsilon = 0.02 * cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour, epsilon, True)

    # Simplify the contour further
    hand_contour_simple = cv2.convexHull(approx)

    defects = cv2.convexityDefects(hand_contour_simple, cv2.convexHull(hand_contour_simple, returnPoints=False))

    if defects is not None:
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            
            # Check if points exist in the approx array
            if s < len(approx) and e < len(approx) and f < len(approx):
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                
                # Angle threshold to consider as a finger
                if angle <= np.pi/2:
                    finger_count += 1

        return finger_count
    else:
        return 0

def gesture_recognition(hand_contour):
    # Check if the hand is open or closed based on finger count
    fingers = count_fingers(hand_contour)
    if fingers >= 3:
        return "open"
    else:
        return "closed"

def main():
    cap = cv2.VideoCapture(0)

    hand_status = "closed"
    previous_hand_status = "closed"

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Find hand contour
        hand_contour = find_hand(frame)

        if hand_contour is not None:
            # Perform hand gesture recognition
            hand_status = gesture_recognition(hand_contour)

            # Print hand status for debugging
            print("Hand Status:", hand_status)

            # Draw hand contour and finger count for visualization
            finger_count = count_fingers(hand_contour)
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam


# In[69]:


labels_all = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (5)\\labels.csv")
print(labels_all.shape)
labels_all.head()


# In[70]:


breed_all = labels_all['breed']
breed_count = breed_all.value_counts()
breed_count.head()


# In[71]:


CLASS_NAME = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog']
labels = labels_all[(labels_all['breed'].isin(CLASS_NAME))]
labels = labels.reset_index()
labels.head()


# In[72]:


X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
# One hot encoding
Y_data = label_binarize(labels['breed'], classes = CLASS_NAME)

# Reading and converting image to numpy array and normalizing dataset
for i in tqdm(range(len(labels))):
    img = image.load_img(os.path.join("C:\\Users\\91701\\Downloads\\archive (5)\\train", '%s.jpg' % labels['id'][i]), target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0

# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))


# In[73]:


model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu', input_shape = (224,224,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', kernel_regularizer = 'l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 16, kernel_size = (7,7), activation ='relu', kernel_regularizer = 'l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 8, kernel_size = (5,5), activation ='relu', kernel_regularizer = 'l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation = "relu", kernel_regularizer = 'l2'))
model.add(Dense(64, activation = "relu", kernel_regularizer = 'l2'))
model.add(Dense(len(CLASS_NAME), activation = "softmax"))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001),metrics=['accuracy'])

model.summary()


# In[74]:


# Splitting the data set into training and testing data sets
X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size = 0.1)
# Splitting the training data set into training and validation data sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size = 0.2)


# In[75]:


epochs = 100
batch_size = 128
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val))


# In[76]:


plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()


# In[77]:


Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# In[80]:


plt.imshow(X_test[1,:,:,:])
plt.show()

# Finding max value from predition list and comaparing original value vs predicted
print("Originally : ",labels['breed'][np.argmax(Y_test[1])])
print("Predicted : ",labels['breed'][np.argmax(Y_pred[1])])


# In[81]:


Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# tqdm is a library in Python that provides a fast, extensible progress bar for
# loops and other iterable processes. When you use from tqdm import tqdm,
# you are importing the tqdm class, which you can then use to wrap an iterable
# and create a progress bar for it. This can be particularly useful when you have
# long-running tasks or loops, as it gives you a visual indication of the progres
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam


# In[3]:


dog_breed_path = "C:\\Users\\91701\\Downloads\\archive (5)\\labels.csv"


# In[4]:


labels_all = pd.read_csv(dog_breed_path)
print(labels_all.shape)
labels_all.head()


# In[5]:


len(labels_all['breed'].unique())


# In[6]:


# Visualize the number of each breeds
breeds_all = labels_all["breed"]
breed_counts = breeds_all.value_counts()
breed_counts


# In[7]:


# Selecting random 50  breeds (Limitation due to computation power)
np.random.seed(1)
CLASS_NAMES =  np.random.choice(labels_all['breed'].unique(), size=50, replace=False)
labels = labels_all[(labels_all['breed'].isin(CLASS_NAMES))]
labels = labels.reset_index()
labels


# In[8]:


labels['breed'].unique()


# In[9]:


len(labels['breed'].unique())


# In[10]:


len(CLASS_NAMES)


# In[12]:


# Creating numpy matrix with zeros
X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
# One hot encoding
Y_data = label_binarize(labels['breed'], classes = CLASS_NAMES)

# Reading and converting image to numpy array and normalizing dataset
for i in tqdm(range(len(labels))):

    # target_size=(224, 224): resizes the image to a standard size of 224x224 pixels.
    # %s for keep 'labels['id'][i]' as string format
    # For example  labels['id'][i] holds the value '00123', the resulting path becomes:
    #         'dog_dataset/train/00123.jpg'
    img = image.load_img(os.path.join("C:\\Users\\91701\\Downloads\\archive (5)\\train", '%s.jpg' % labels['id'][i]), target_size=(224, 224))
    img = image.img_to_array(img)

    x = np.expand_dims(img.copy(), axis=0)
    # Normalizing pixel values  to avoid the possibility of exploding gradients
    # because of the high range of the pixels [0, 255], and improve the convergence speed.
    X_data[i] = x / 255.0

# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))


# In[93]:


Y_data = label_binarize(labels['breed'], classes = list(labels['breed'].unique()))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))


# In[94]:


# Splitting the data set into training and testing data sets
X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size = 0.1)
# Splitting the training data set into training and validation data sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size = 0.2)


# In[95]:


from keras.applications import MobileNetV2
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# Load pre-trained MobileNetV2 model
# include_top: whether to include the 3 fully-connected layers at the top of the network.
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=True, 
    input_shape=(224, 224, 3), 
    pooling='max',
    classifier_activation="softmax"
)

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the MobileNetV2 base
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(724, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(Y_data.shape[1], activation='softmax'))  # Adjust the number of classes based on your dataset

# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001),metrics=['accuracy'])

model.summary()


# In[1]:


epochs = 70
batch_size = 192

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                    validation_data = (X_val, Y_val))


# In[ ]:


# Plot the training history
plt.figure(figsize=(6, 3))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()


# In[ ]:


plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()


# In[ ]:


Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# In[ ]:


plt.imshow(X_test[1,:,:,:])
plt.show()

# Finding max value from predition list and comaparing original value vs predicted
print("Originally : ",labels['breed'][np.argmax(Y_test[1])])
print("Predicted : ",labels['breed'][np.argmax(Y_pred[1])])

