import os
import glob
from ultralytics import YOLO

weights_path = 'runs/detect/yolo_documents/pilot_table_detection/weights/best.pt'
images_dir   = 'images/val'
model = YOLO(weights_path)

image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))[:3]
with open('debug_out.txt', 'w') as f:
    for img_path in image_paths:
        f.write(f"\n--- {img_path} ---\n")
        results = model.predict(img_path, conf=0.01, verbose=False) # very low conf to see all
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            bbox = [round(x, 2) for x in box.xyxy[0].tolist()]
            f.write(f"Class: {cls_id}, Conf: {conf:.2f}, Box: {bbox}\n")
