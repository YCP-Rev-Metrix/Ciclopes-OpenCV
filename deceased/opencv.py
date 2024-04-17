import cv2

# Load the video
video_path = 'SimVideos/testReal.mp4'
cap = cv2.VideoCapture(video_path)

# Define the lower and upper boundaries for detecting red color in HSV space
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)

# Define the region of interest (bottom and middle of the screen)
roi_width = 100  # Width of the region of interest
roi_height = 100  # Height of the region of interest
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x = int((frame_width - roi_width) / 2)  # Starting x-coordinate
roi_y = frame_height - roi_height - 50  # Starting y-coordinate, moved up by 50 pixels

# Loop through the frames3
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_y -= 5

    # Define the region of interest
    roi = hsv[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(roi, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set detection status text color
    if contours:
        detection_status_color = (0, 255, 0)  # Green color for detection
        detection_status = "Ball detected"
    else:
        detection_status_color = (0, 0, 255)  # Red color for no detection
        detection_status = "No ball detected"

    # Draw bounding box around the ball if detected
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), (0, 255, 0), 2)

    # Draw the search area
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
