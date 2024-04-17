import cv2
import numpy as np

def adjust_roi(x, y, radius, frame_width, frame_height, padding=10):
    """
    Adjust the ROI based on the ball's position and size.
    Add some padding to ensure the ball is always within the ROI.
    """
    roi_x = max(0, int(x - radius - padding))
    roi_y = max(0, int(y - radius - padding))
    roi_width = min(frame_width - roi_x, int(2 * radius + 2 * padding))
    roi_height = min(frame_height - roi_y, int(2 * radius + 2 * padding))
    return roi_x, roi_y, roi_width, roi_height

# Initialize video capture object
cap = cv2.VideoCapture('SimVideos/testReal.mp4')

# Initialization for the ROI
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_x, roi_y, roi_width, roi_height = 0, 0, frame_width, frame_height

# Define minimum and maximum radius and circularity threshold for detected circles
min_radius = 30  # Set the minimum radius based on your requirements
max_radius = 200  # Maximum radius can be adjusted as needed
min_circularity = 0.7  # Adjust this threshold to capture near-circular objects

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Convert ROI to grayscale and apply Gaussian blur
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    # Thresholding or color filtering can be applied here if necessary
    _, thresh = cv2.threshold(blurred_roi, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded ROI
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 100 and perimeter > 0:  # Minimum area threshold and avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= min_circularity:  # Check for circularity
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if min_radius <= radius <= max_radius:
                    center = (int(x), int(y))
                    radius = int(radius)

                    # Draw the circle and center of the ball on the original frame
                    cv2.circle(frame, (roi_x + center[0], roi_y + center[1]), radius, (0, 255, 0), 2)
                    cv2.circle(frame, (roi_x + center[0], roi_y + center[1]), 2, (0, 0, 255), 3)

                    # Adjust ROI based on the ball's new position and size
                    roi_x, roi_y, roi_width, roi_height = adjust_roi(roi_x + x, roi_y + y, radius, frame_width, frame_height)

    # Display the frame with detected ball and ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    # Quit if 'Q' or 'q' is pressed
    if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
