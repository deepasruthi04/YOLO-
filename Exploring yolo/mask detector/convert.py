from ultralytics import YOLO
import cv2
import cvzone
from collections import Counter

# Load the trained YOLO model
model_path = r"C:\Users\deepasruthi.ramesh\Documents\Image_detection\VI_Learning\MaskDetector\runs\detect\train10\weights\best.pt"
model = YOLO(model_path)

# Path to the image you want to predict
image_path = r"C:\Users\deepasruthi.ramesh\Documents\Image_detection\VI_Learning\MaskDetector\dataset\Screenshot (64).png"

# Read the image
frame = cv2.imread(image_path)
if frame is None:
    print("❌ Failed to load image. Check the image path.")
    exit()

# Run YOLO prediction on the image
results = model.predict(source=frame, stream=False, verbose=False)
print(results)
object_list = []

# Loop through results and draw boxes
for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        # ✅ Print detections in console
        print(f"Detected: {class_name} ({conf:.2f})")

        # Draw bounding boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if class_name == "mask" else (0, 0, 255)

        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=3, colorR=color)
        cvzone.putTextRect(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), scale=1, thickness=1)
        object_list.append(class_name)

# Show counts of detected classes
counts = Counter(object_list)
y_offset = 40
for obj, cnt in counts.items():
    cv2.putText(frame, f"{obj}: {cnt}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_offset += 30

# Display the image with detections
cv2.imshow("Mask Detector - Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
