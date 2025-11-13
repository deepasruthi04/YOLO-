from ultralytics import YOLO
import cv2
import cvzone
from collections import Counter


model_path = model = YOLO(r"C:\Users\deepasruthi.ramesh\Documents\Image_detection\VI_Learning\ObjCountCustom\runs\detect\train17\weights\best.pt")

model = YOLO(model_path)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open stream. Check camera URL/credentials or webcam ID.")
    exit()

print("Stream opened successfully. Press 'Q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame. Check camera connection.")
        break
    
    results = model.predict(source=frame, stream=False, verbose=False, conf=0.5) or []
    object_list = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=3)
                cvzone.putTextRect(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), scale=1, thickness=1)
                object_list.append(class_name)

   
    counts = Counter(object_list)
    y_offset = 40
    for obj, cnt in counts.items():
        cv2.putText(frame, f"{obj}: {cnt}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    
    cv2.imshow("Object Counter", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
