from ultralytics import YOLO
import cv2, torch, cvzone

device = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture("chapters/videos/cars.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

model = YOLO("yolo-weights/yolov8n.pt").to(device)

mask = cv2.imread("projects/car-counter/mask.png")

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, img, mask)
    
    if not success:
        print("Failed to read frame from video")
        break
    
    results = model(img, stream=True)
    
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
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)
                cvzone.putTextRect(img, f"{conf} {label}", (max(0, x1), max(10, y1)), scale=1, thickness=1, offset=3)
            
    cv2.imshow("Mask", imgRegion) 
    cv2.imshow("Image", img)
    # if cv2.waitKey(1) == ord('q'):
    #     break\
        
    cv2.waitKey(0)
    
cap.release()
cv2.destroyAllWindows()