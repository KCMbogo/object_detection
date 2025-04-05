from ultralytics import YOLO
import cv2, cvzone, torch, yaml

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# cap = cv2.VideoCapture(0)
# # cap.set(propId, value)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("chapters/videos/ppe-3-1.mp4")

model = YOLO("projects/ppe detection/ppe.pt").to(device)

with open('ConstructionSiteSafety/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
    
class_names = data['names']

detection_color = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # For cv2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            w, h = x2-x1, y2-y1
                        
            # confidente
            conf = round(float(box.conf[0]), 2)
            
            # class name
            cls = int(box.cls[0])
            label = class_names[cls]
            
            if label in ['Hardhat', 'Mask', 'Safety Vest', 'Safety Cone', 'Gloves'] and conf > 0.5:
                detection_color = (0, 255, 0)
            else:
                detection_color = (0, 0, 255)
                
            if label != 'Person':
                cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=detection_color)
                cvzone.putTextRect(img, f'{label}', (max(0, x1), max(10, y1)), scale=1, thickness=1)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()