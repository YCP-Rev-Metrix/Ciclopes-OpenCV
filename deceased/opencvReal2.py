import cv2
import numpy as np

# Load the video
video_path = 'SimVideos/testReal.mp4'
cap = cv2.VideoCapture(video_path)

# Set minimum and maximum contour area thresholds
min_contour_area = 85
max_contour_area = 140  # Adjust this value to limit the maximum detection area

# Initial movement increments
move_increment_x = 18
move_increment_y = 35

# Function to check if contour is approximately round
def is_round_contour(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return len(approx) > 6

# Define the region of interest (ROI)
roi_width = 150  # Width of the region of interest
roi_height = 150  # Height of the region of interest
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2) - 200  # Starting x-coordinate, shifted to the left by 100 pixels
roi_y = frame_height - roi_height - 100  # Starting y-coordinate, moved up by 50 pixels

# Loop through the frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Define ROI
    roi_x_start = max(0, roi_x)
    roi_y_start = max(0, roi_y)
    roi_x_end = min(frame_width, roi_x + roi_width)
    roi_y_end = min(frame_height, roi_y + roi_height)

    # Define ROI
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set detection status text color
    round_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area and is_round_contour(cnt)]
    if round_contours:
        detection_status_color = (0, 255, 0)  # Green color for detection
        detection_status = "Round object detected"
        # Get the bounding box of the largest contour (assumed to be the ball)
        x, y, w, h = cv2.boundingRect(max(round_contours, key=cv2.contourArea))
    else:
        detection_status_color = (0, 0, 255)  # Red color for no detection
        detection_status = "No round object detected"

    # Draw bounding box around the detected objects
    for contour in round_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the region of interest (ROI) in blue
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # Display detection status on screen
    cv2.putText(frame, detection_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, detection_status_color, 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
