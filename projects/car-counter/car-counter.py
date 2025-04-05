from ultralytics import YOLO
from sort import *

import cv2, torch, cvzone
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

model = YOLO("yolo-weights/yolov8l.pt").to(device)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("chapters/videos/cars.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mask = cv2.imread("projects/car-counter/mask.png")

line_start = (400, 297)
line_end = (673, 297)
line_color = (255, 0, 0)
line_color_detect = (0, 255, 0)

total_counts = []

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to read frame from video")
        break
    
    graphics = cv2.imread("projects/car-counter/graphics.png", cv2.IMREAD_UNCHANGED)

    img = cvzone.overlayPNG(img, graphics, (0, 0))
    
    cv2.line(img, line_start, line_end, line_color, 3)
    
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)
    
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label in ['car', 'truck', 'motorbike', 'bus'] and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
                # cvzone.putTextRect(img, f"{conf} {label}", (max(0, x1), max(10, y1)), scale=1, thickness=1, offset=3)
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))
                
    detections = np.array(detections)
    
    tracks = tracker.update(detections)
    
    for track in tracks:
        x1, y1, x2, y2, id = map(int, track)
        w, h = x2-x1, y2-y1
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
        cvzone.putTextRect(img, f"{id}", (max(0, x1), max(10, y1)), scale=2, thickness=3, offset=10)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)
        
        if line_start[0] < cx < line_end[0] and (line_start[1]-15) < cy < (line_end[1]+15):
            if total_counts.count(id) == 0:
                total_counts.append(id)
                cv2.line(img, line_start, line_end, line_color_detect, 3)

        # cvzone.putTextRect(img, f"Count: {len(total_counts)}", (50, 50), scale=2, thickness=3, offset=10)
        cvzone.putTextRect(img=img, text=str(len(total_counts)), pos=(255, 100), font=cv2.FONT_HERSHEY_PLAIN, scale=5, colorT=(255, 255, 255), thickness=8)
    
        
        
            
    # cv2.imshow("Mask", imgRegion) 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    # cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()