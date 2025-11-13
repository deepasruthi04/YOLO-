from ultralytics import YOLO  # type: ignore
import cv2
import cvzone
from collections import defaultdict
import supervision as sv
import numpy as np

# Load YOLO model (replace with your custom model path if needed)
model = YOLO("yolov8n.pt")

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Start webcam (0) or RTSP stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()
else:
    print("✅ Camera connected")

# Unique object count storage
object_counts = defaultdict(set)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Run YOLO detection
    results = model(frame)

    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Track detections (assigns unique IDs)
    tracked = tracker.update_with_detections(detections)

    # Ensure arrays are not None (for type safety)
    if (
        tracked.xyxy is None
        or tracked.class_id is None
        or tracked.confidence is None
        or tracked.tracker_id is None
    ):
        continue

    # Iterate through all tracked objects
    for box, cls_id, conf, track_id in zip(
        tracked.xyxy,
        tracked.class_id,
        tracked.confidence,
        tracked.tracker_id
    ):
        if track_id is None:
            continue

        # Convert bbox to int
        x1, y1, x2, y2 = map(int, box)

        # Get class name
        class_name = model.names.get(int(cls_id), "Unknown")

        # Store unique IDs for counting
        object_counts[class_name].add(int(track_id))

        # Draw the box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            frame,
            f"{class_name} ID:{track_id}",
            (x1, max(30, y1 - 10)),
            scale=1,
            thickness=1,
            colorR=(255, 255, 255)
        )

    # Draw live counts
    y_offset = 30
    for cls_name, ids in object_counts.items():
        cv2.putText(
            frame,
            f"{cls_name}: {len(ids)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        y_offset += 30

    # Show frame
    cv2.imshow("Object Counter", frame)

    # Quit on "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
