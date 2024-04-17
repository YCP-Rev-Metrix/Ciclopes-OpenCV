import cv2
import numpy as np

# Load the video
video_path = 'SimVideos/testReal.mp4'
cap = cv2.VideoCapture(video_path)

# Define the parameters for the ball detection
# Red color range in HSV
ball_color_lower = np.array([0, 50, 50])  # Lower bound for red color
ball_color_upper = np.array([10, 255, 255])  # Upper bound for red color
min_radius = 10
max_radius = 200

# Initialize variables for ROI tracking
roi_width = 200
roi_height = 200
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2) - 200
roi_y = frame_height - roi_height - 100  # Move ROI to the bottom middle

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the video has ended
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, ball_color_lower, ball_color_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Get the largest contour (assumed to be the ball)
        ball_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(ball_contour)

        # Proceed if the detected contour has a radius within the specified range
        if min_radius < radius < max_radius:
            # Draw a green bounding box around the detected ball
            cv2.rectangle(frame, (int(x - radius), int(y - radius)), (int(x + radius), int(y + radius)), (0, 255, 0), 2)

            # Update ROI position based on ball position
            roi_x = int(x - roi_width / 2)
            roi_y = int(y - roi_height / 2)

            # Ensure ROI stays within frame boundaries
            roi_x = max(0, min(roi_x, frame_width - roi_width))
            roi_y = max(0, min(roi_y, frame_height - roi_height))

    # Draw a blue bounding box around the ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Bowling Ball Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
