from ultralytics import YOLO
from sort import *
import torch, cv2, cvzone

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

model = YOLO("yolo-weights/yolov8l.pt").to(device)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("chapters/videos/people.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mask = cv2.imread("projects/people-counter/mask.png")

line_start = (223, 388)
line_end = (589, 290)
line_color = (255, 0, 0)
line_color_detect = (0, 255, 0)

down_counts = []
up_counts = []
        
track_history = {}

# Helper for calculating distance from point to line
def is_point_near_line(px, py, x1, y1, x2, y2, threshold=10):
    # Compute distance from point (px, py) to line (x1, y1)-(x2, y2)
    line_mag = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    if line_mag < 1e-6:
        return False
    u = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / (line_mag**2)
    if u < 0 or u > 1:
        return False
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    dist = ((px - ix)**2 + (py - iy)**2)**0.5
    return dist < threshold


while True:
    ret, img = cap.read()
     
    if not ret:
        print("Failed to read frame")
        break
    
    graphics = cv2.imread("projects/people-counter/graphics.png", cv2.IMREAD_UNCHANGED)
    
    img = cvzone.overlayPNG(img, graphics, (0, 0))
    
    cv2.line(img=img, pt1=line_start, pt2=line_end, color=line_color, thickness=2)
    
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img_region = cv2.bitwise_and(img, mask)
    
    results = model(img_region, stream=True)
    
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1
            
            cls = int(box.cls[0])
            label = model.names[cls]
            
            conf = round(float(box.conf[0]), 2)
            
            if label == "person" and conf > 0.5:
                people = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, people))

    tracks = tracker.update(detections)
    
    for track in tracks:
        x1, y1, x2, y2, id = map(int, track)
        w, h = x2-x1, y2-y1
            
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img=img, center=(cx, cy), radius=3, color=(255, 0, 0), thickness=cv2.FILLED)

        direction = None
        
        if is_point_near_line(cx, cy, *line_start, *line_end, threshold=10):
            if id in track_history:
                prev_cy = track_history[id]
                if cy > prev_cy:
                    direction = "down"
                elif cy < prev_cy:
                    direction = "up"
                else:
                    direction = "stationary"
                
                if up_counts.count(id) == 0 and direction == "up":
                    up_counts.append(id)
                elif down_counts.count(id) == 0 and direction == "down":
                    down_counts.append(id)
                    
                cv2.line(img=img, pt1=line_start, pt2=line_end, color=line_color_detect, thickness=2)
            
        track_history[id] = cy
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3)
        # cvzone .putTextRect(img, f"{id}", (max(0, x1), max(10, y1)), scale=3, thickness=2, offset=3)
        
    cvzone.putTextRect(img=img, text=str(len(up_counts)), pos=(210, 85), scale=2, thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX, offset=3)
    cvzone.putTextRect(img=img, text=str(len(down_counts)), pos=(465, 85), scale=2, thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX, offset=3)
    
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    