import cv2
import numpy as np

# Initialize video capture object for reading the video from a specified file
cap = cv2.VideoCapture('SimVideos/testReal.mp4')

# Initialize the background subtractor object for foreground-background segmentation
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Define font type and text parameters for on-screen text display
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
font_scale = 1
color_not_detected = (0, 0, 255)  # Red color for "Object Not Detected" text
color_detected = (0, 255, 0)      # Green color for "Object Detected" text
thickness = 2

# Initialize parameters for defining the region of interest (ROI) within the video frame
roi_width = 150
roi_height = 150
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2) - 195
roi_y = frame_height - roi_height - 100  # Set ROI to the specified location within the frame

# Set thresholds for the object area to filter out noise and small irrelevant objects
min_object_area = 50
max_object_area = 5000

# Loop through each frame in the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is captured
    
    # Convert the captured frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction to isolate moving objects from the background
    fg_mask = bg_subtractor.apply(gray)
    
    # Find contours in the foreground mask to identify individual objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Flag to indicate if any object is detected in the current frame
    object_detected = False
    
    # Iterate through each contour found in the frame
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Only consider contours that fall within the specified area range
        if min_object_area < area < max_object_area:
            # Approximate contour shape to a simpler polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Detect circularity to differentiate objects of interest
            if len(approx) >= 4:  # Check for complex shapes
                (x, y), radius = cv2.minEnclosingCircle(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                # Consider the object only if it has a significant circularity
                if circularity > 0.4:
                    # Mark the detected object with a green circle and center point
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(frame, center, radius, color_detected, 2)
                    cv2.circle(frame, center, 2, color_detected, -1)  # Center point
                    object_detected = True
                    
                    # Dynamically update the ROI based on object movement
                    roi_height = roi_height - 2
                    roi_width = roi_width - 2
                    roi_x = int(x - roi_width / 2)
                    roi_y = int(y - roi_height / 2)
                    roi_x = max(0, min(roi_x, frame_width - roi_width))  # Ensure ROI stays within frame
                    roi_y = max(0, min(roi_y, frame_height - roi_height))

    # Display the detection status on the frame
    if object_detected:
        cv2.putText(frame, "Object Detected", org, font, font_scale, color_detected, thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Object Not Detected", org, font, font_scale, color_not_detected, thickness, cv2.LINE_AA)
    
    # Highlight the ROI on the frame for visual reference
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
    
    # Show the frame with applied annotations
    cv2.imshow('Frame', frame)
    
    # Allow the user to quit the video processing by pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows after finishing the processing
