import cv2

# Load the video
video_path = 'SimVideos/testReal.mp4'
cap = cv2.VideoCapture(video_path)

# Define the lower and upper boundaries for detecting red, white, and black colors in HSV space
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
lower_white = (0, 0, 200)
upper_white = (180, 30, 255)
lower_black = (0, 0, 0)
upper_black = (180, 255, 30)

# Define the region of interest (bottom and middle of the screen)
roi_width = 150  # Width of the region of interest
roi_height = 150  # Height of the region of interest
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2) - 200 # Starting x-coordinate, shifted to the left by 100 pixels
roi_y = frame_height - roi_height - 100  # Starting y-coordinate, moved up by 50 pixels

# Set minimum contour area threshold
min_contour_area = 0

# Delay before ROI movement starts (in frames)
delay_frames = 30
frame_count = 0

# Initial movement increments
move_increment_x = 18
move_increment_y = 35

# Loop through the frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the region of interest
    roi = hsv[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Threshold the HSV image to get masks for red, white, and black colors
    mask_red = cv2.inRange(roi, lower_red, upper_red)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set detection status text color
    if contours:
        detection_status_color = (0, 255, 0)  # Green color for detection
        detection_status = "Object detected"
    else:
        detection_status_color = (0, 0, 255)  # Red color for no detection
        detection_status = "No object detected"

    # Draw bounding box around the detected objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), (0, 255, 0), 2)

    # Draw the search area before delay frames
    if frame_count <= delay_frames:
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # Display detection status on screen
    cv2.putText(frame, detection_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, detection_status_color, 2)

    # Increment frame count
    frame_count += 1

    # If the delay frames haven't passed yet, continue to the next frame without moving ROI
    if frame_count <= delay_frames:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    # Slow down movement increments as frame count increases
    move_increment_x -= 0.25
    move_increment_y -= 0.5

    # Move ROI to the right and up after the delay frames
    roi_x += int(move_increment_x)  # Move ROI to the right
    roi_y -= int(move_increment_y)  # Move ROI up

    # Draw the search area after delay frames
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
