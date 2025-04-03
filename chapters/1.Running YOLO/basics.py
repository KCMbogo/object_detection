from ultralytics import YOLO
import cv2

# Load pretrained model
model = YOLO("../../yolo-weights/yolov8n.pt")
results = model("chapters/1.Running YOLO/images/1.jpg", show=True)
cv2.waitKey(0) # unless the user inputs don't do anything