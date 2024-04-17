import cv2
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture('SimVideos/testReal.mp4')

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize font and text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
font_scale = 1
color_not_detected = (0, 0, 255)  # Red color for "Object Not Detected"
color_detected = (0, 255, 0)      # Green color for "Object Detected"
thickness = 2

# Define the region of interest (ROI) parameters
roi_width = 150#105q
roi_height = 150#105
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2)-195
roi_y = frame_height - roi_height -100 # Move ROI to the bottom middle

# Define minimum and maximum object area thresholds
min_object_area = 50
max_object_area = 5000

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction to detect moving objects
    fg_mask = bg_subtractor.apply(gray)
    
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_detected = False
    
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)
        
        # Check if the contour area is within the specified range
        if min_object_area < area < max_object_area:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the contour is close to a circle (by comparing its perimeter and area)
            if len(approx) >= 4:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter)

                # Check if the contour is circular
                if circularity > 0.4:
                    # Draw the bounding circle around the detected object in green
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)
                    cv2.circle(frame, center, 2, (0, 255, 0), -1)  # Draw center point
                    object_detected = True
                    
                    # Update ROI position based on the detected object's position
                    roi_height = roi_height-2
                    roi_width = roi_width-2
                    roi_x = int(x - roi_width / 2)
                    roi_y = int(y - roi_height / 2)
                    roi_x = max(0, min(roi_x, frame_width - roi_width))
                    roi_y = max(0, min(roi_y, frame_height - roi_height))
    
    # Display object detection status text
    if object_detected:
        cv2.putText(frame, "Object Detected", org, font, font_scale, color_detected, thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Object Not Detected", org, font, font_scale, color_not_detected, thickness, cv2.LINE_AA)
    
    # Draw ROI rectangle in blue
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
