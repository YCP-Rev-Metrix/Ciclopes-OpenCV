import cv2
import numpy as np
import pandas as pd

# Initialize video capture object
cap = cv2.VideoCapture('SimVideos/curveedit.mp4')  # Change file path as needed

# Define the lower and upper bounds of the red color in HSV (two ranges)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])

# Define font and text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
font_scale = 1
color_detected = (0, 255, 0)  # Green for text
color_not_detected = (0, 0, 255)  # Red for not detected
thickness = 2

# Blue color for the ROI
roi_color = (255, 0, 0)

paused = False  # Video playback state flag
skippedFrame = False

# Initialize a list to store coordinates
coordinates = []
show_dots = False  # Flag to show/hide red dots

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Processing the frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_detected = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 2:
                cv2.circle(frame, (int(x), int(y)), int(radius), color_detected, 2)
                cv2.circle(frame, (int(x), int(y)), 5, color_detected, -1)
                ball_detected = True

                # Append the coordinates to the list
                coordinates.append((x, y))

                roi_margin = int(radius * 0.3)
                roi_x1 = max(int(x - radius - roi_margin), 0)
                roi_y1 = max(int(y - radius - roi_margin), 0)
                roi_x2 = min(int(x + radius + roi_margin), frame.shape[1])
                roi_y2 = min(int(y + radius + roi_margin), frame.shape[0])
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)

        if ball_detected:
            cv2.putText(frame, "Ball Detected", org, font, font_scale, color_detected, thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Ball Not Detected", org, font, font_scale, color_not_detected, thickness, cv2.LINE_AA)

    # Plot red dots if the flag is set
    if show_dots:
        for (x, y) in coordinates:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)

    if skippedFrame:
        paused = True
        skippedFrame = False
    if paused:
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == ord(' '):  # If space is pressed, step through frames
            paused = False  # Keep the video paused state
            skippedFrame = True
            continue
        elif key == ord('r'):
            paused = False  # Resume video 
        elif key == ord('g'):
            show_dots = not show_dots
    else:
        key = cv2.waitKey(30) & 0xFF  # Non-paused regular playback wait time
        if key == ord('p'):
            paused = True
        elif key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save coordinates to an Excel file using pandas
df = pd.DataFrame(coordinates, columns=['X', 'Y'])
df.to_excel('output_coordinates.xlsx', index=False)