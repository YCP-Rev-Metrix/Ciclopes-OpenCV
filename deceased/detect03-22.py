import cv2
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture('SimVideos/testReal.mp4')

# Define font and text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
font_scale = 1
color_detected = (0, 255, 0)  # Green color for text
thickness = 2

# Blue color for the ROI
roi_color = (255, 0, 0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Define the center point for the ROI at the bottom middle of the screen
    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width//2 - 200
    center_y = frame_height//2 + 350 # Assuming the ROI height is 150

    # Calculate the ROI coordinates based on the center point
    roi_width = 150
    roi_height = 150
    roi_x1 = center_x - roi_width // 2
    roi_y1 = center_y - roi_height // 2
    roi_x2 = center_x + roi_width // 2
    roi_y2 = center_y + roi_height // 2

    # Draw ROI rectangle in blue
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)

    # Convert the captured frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find contours in the gray image
    contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_detected = False
    
    # Only proceed if at least one contour was found
    if contours:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Only proceed if the radius meets a minimum size threshold and the center is within the ROI
        if radius > 2 : #and roi_x1 < x < roi_x2 and roi_y1 < y < roi_y2
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), color_detected, 2)
            cv2.circle(frame, (int(x), int(y)), 5, color_detected, -1)
            ball_detected = True

    # Display the ball detection status
    if ball_detected:
        cv2.putText(frame, "Ball Detected", org, font, font_scale, color_detected, thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Ball Not Detected", org, font, font_scale, color_detected, thickness, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit loop if 'q' or 'Q' is pressed
    if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q')]:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
