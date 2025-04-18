from ultralytics import YOLO
import cv2, cvzone, torch

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device}")

# cap = cv2.VideoCapture(0)
# # cap.set(propId, value)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("chapters/videos/ppe-1-1.mp4")

model = YOLO("yolo-weights/yolov8n.pt").to(device)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # For cv2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # For cvzone
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # confidente
            conf = round(float(box.conf[0]), 2)
            
            # class name
            cls = int(box.cls[0])
            label = model.names[cls]
            
            cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(10, y1)), scale=0.8, thickness=1)
            
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)