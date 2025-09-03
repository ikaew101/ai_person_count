import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py is in the same directory

# Define the coordinates of the lines based on the provided image
# IMPORTANT: You need to manually adjust these coordinates based on your video's resolution.
# (x1, y1), (x2, y2)
red_line = [(616, 275), (1345, 279)]
blue_line = [(616, 275), (552, 983)]
green_line = [(1345, 279), (1537, 956)]
yellow_line = [(552, 983), (1537, 956)]

# Set the desired display window size
# You can change these values to adjust the window size
display_width = 1080
display_height = 740

# A helper function to check if a point has crossed a line segment
def is_crossing_line(p1, p2, line_start, line_end):
    # Determine the side of the line for each point
    side1 = np.sign(
        (line_end[0] - line_start[0]) * (p1[1] - line_start[1]) -
        (line_end[1] - line_start[1]) * (p1[0] - line_start[0])
    )
    side2 = np.sign(
        (line_end[0] - line_start[0]) * (p2[1] - line_start[1]) -
        (line_end[1] - line_start[1]) * (p2[0] - line_start[0])
    )
    # Check if the points are on different sides and the line segments intersect
    return side1 != side2 and side1 != 0 and side2 != 0

# Load the video file
# Change this path to your video file's location.
video_path = 'vdo_test1.mp4'

# Load the YOLOv8 model for person detection
model = YOLO('yolov8n.pt')

# Initialize the SORT tracker
tracker = Sort()

# Counters and state for each person
total_counts = {'right': 0, 'left': 0, 'straight': 0, 'inbound': 0, 'outbound': 0}
person_states = {}  # Stores state of each person: {'crossed_red': bool, 'start_time': float, 'destination_line': str, 'last_pos': tuple}

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_width / original_height
if display_width / display_height > aspect_ratio:
    display_width = int(display_height * aspect_ratio)
else:
    display_height = int(display_width / aspect_ratio)

# Mouse callback function to get coordinates
def draw_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        frame_copy = param[0].copy()
        
        # Scale coordinates for mouse position
        scale_x = original_width / display_width
        scale_y = original_height / display_height
        
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        
        cv2.putText(frame_copy, f'Original X: {orig_x}, Y: {orig_y}', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Video Analysis', frame_copy)

# Main loop to process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for display
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Set up the window and mouse callback
    cv2.namedWindow('Video Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Analysis', display_width, display_height)
    cv2.setMouseCallback('Video Analysis', draw_coordinates, param=[frame])

    # 1. Person Detection using YOLO
    results = model(frame, stream=True)
    detections = []
    for r in results:
        # Filter detections to only include 'person' class (class_id=0 for COCO dataset)
        for box in r.boxes.data:
            # Added a check for the number of elements in the box to prevent index errors
            if len(box) >= 6:
                x1, y1, x2, y2, score, class_id = box
                if int(class_id) == 0:  # Class '0' is 'person' in COCO dataset
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])

    # 2. Object Tracking using SORT
    # Added a check to ensure detections list is not empty before updating tracker
    if detections:
        tracked_objects = tracker.update(np.array(detections))
    else:
        # If no detections, update tracker with an empty array to maintain state
        tracked_objects = tracker.update(np.empty((0, 5)))

    # Get a list of current object IDs to check for missing IDs
    current_ids = [int(t[4]) for t in tracked_objects]
    
    # Clean up states for objects that are no longer tracked
    ids_to_remove = [obj_id for obj_id in person_states if obj_id not in current_ids]
    for obj_id in ids_to_remove:
        del person_states[obj_id]

    # 3. Draw lines and process tracking
    cv2.line(frame, red_line[0], red_line[1], (0, 165, 255), 2)
    cv2.line(frame, blue_line[0], blue_line[1], (0, 0, 0), 2)
    cv2.line(frame, green_line[0], green_line[1], (128, 0, 128), 2)
    cv2.line(frame, yellow_line[0], yellow_line[1], (0, 100, 0), 2)

    for track in tracked_objects:
        x1, y1, x2, y2, obj_id = track
        
        # Calculate centroid
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Initialize state if it's a new person
        if obj_id not in person_states:
            person_states[obj_id] = {
                'crossed_red': False,
                'start_time': 0,
                'destination_line': None,
                'last_pos': centroid
            }
            
        last_pos = person_states[obj_id]['last_pos']
        current_pos = centroid
        
        # Check for inbound/outbound crossing
        if is_crossing_line(last_pos, current_pos, red_line[0], red_line[1]):
            # Check direction of movement based on y-coordinate relative to the line
            if current_pos[1] > last_pos[1]:  # Moving from top to bottom (inbound)
                if not person_states[obj_id]['crossed_red']:
                    person_states[obj_id]['crossed_red'] = True
                    person_states[obj_id]['start_time'] = time.time()
                    total_counts['inbound'] += 1
                    print(f"Person {int(obj_id)} entered (inbound). Total inbound: {total_counts['inbound']}")
            else:  # Moving from bottom to top (outbound)
                if not person_states[obj_id]['crossed_red']:
                    person_states[obj_id]['crossed_red'] = True
                    person_states[obj_id]['start_time'] = time.time()
                    total_counts['outbound'] += 1
                    print(f"Person {int(obj_id)} entered (outbound). Total outbound: {total_counts['outbound']}")
                    person_states[obj_id]['destination_line'] = None

        # Check for destination line crossings if the red line was crossed and destination is not set
        if person_states[obj_id]['crossed_red'] and person_states[obj_id]['destination_line'] is None:
            if is_crossing_line(last_pos, current_pos, blue_line[0], blue_line[1]):
                person_states[obj_id]['destination_line'] = 'right'
                duration = time.time() - person_states[obj_id]['start_time']
                total_counts['right'] += 1
                print(f"Person {int(obj_id)} turned right. Duration: {duration:.2f}s. Total right: {total_counts['right']}")

            elif is_crossing_line(last_pos, current_pos, green_line[0], green_line[1]):
                person_states[obj_id]['destination_line'] = 'left'
                duration = time.time() - person_states[obj_id]['start_time']
                total_counts['left'] += 1
                print(f"Person {int(obj_id)} turned left. Duration: {duration:.2f}s. Total left: {total_counts['left']}")

            elif is_crossing_line(last_pos, current_pos, yellow_line[0], yellow_line[1]):
                person_states[obj_id]['destination_line'] = 'straight'
                duration = time.time() - person_states[obj_id]['start_time']
                total_counts['straight'] += 1
                print(f"Person {int(obj_id)} went straight. Duration: {duration:.2f}s. Total straight: {total_counts['straight']}")


        # Update last position for the next frame
        person_states[obj_id]['last_pos'] = current_pos
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, centroid, 5, (0, 255, 255), -1)


    # Display the results on the frame
    cv2.putText(frame, f"Right: {total_counts['right']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Left: {total_counts['left']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Straight: {total_counts['straight']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Inbound: {total_counts['inbound']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Outbound: {total_counts['outbound']}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Video Analysis', frame)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()