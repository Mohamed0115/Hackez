# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install ultralytics paddlepaddle-gpu paddleocr==2.8.1
# !pip install "langchain>=0.3.0,<0.4.0" "langchain-community>=0.3.0,<0.4.0" "langchain-core>=0.3.31,<0.4.0"

# ============================================================
# CELL 2: Mount Google Drive (if your weights/images are on Drive)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# CELL 3: Set your paths
# ============================================================
import os

# --- EDIT THESE ---
weights_path = 'runs/detect/yolo_documents/pilot_table_detection/weights/best.pt'
images_dir   = 'ocr_images' # Update to the Phase 2 test images directory
output_json  = 'final_submission.json'
# ------------------

# ============================================================
# CELL 4: Helper — formulate cells and run OCR
# ============================================================
def extract_cells_and_ocr(table_bbox, rows, columns, img, ocr_model):
    cells = []
    
    if not rows and not columns:
        rows = [table_bbox]
        columns = [table_bbox]
    elif not rows:
        rows = [table_bbox]
    elif not columns:
        columns = [table_bbox]
        
    rows = sorted(rows, key=lambda r: (r[1] + r[3]) / 2)
    columns = sorted(columns, key=lambda c: (c[0] + c[2]) / 2)
    
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(columns):
            cx1, cy1, cx2, cy2 = col
            rx1, ry1, rx2, ry2 = row
            
            x1 = max(cx1, table_bbox[0])
            y1 = max(ry1, table_bbox[1])
            x2 = min(cx2, table_bbox[2])
            y2 = min(ry2, table_bbox[3])
            
            if x2 > x1 and y2 > y1:
                cell_box = [x1, y1, x2, y2]
                
                # --- CALCULATE SPANNING ---
                rowspan, colspan = 0, 0
                for r in rows:
                    r_overlap = max(0, min(y2, r[3]) - max(y1, r[1]))
                    if r_overlap / max(1e-5, r[3] - r[1]) > 0.5: rowspan += 1
                for c in columns:
                    c_overlap = max(0, min(x2, c[2]) - max(x1, c[0]))
                    if c_overlap / max(1e-5, c[2] - c[0]) > 0.5: colspan += 1
                
                # --- PADDLE OCR TEXT EXTRACTION ---
                img_h, img_w = img.shape[:2]
                crop_y1 = max(0, int(y1))
                crop_y2 = min(img_h, int(y2))
                crop_x1 = max(0, int(x1))
                crop_x2 = min(img_w, int(x2))
                
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                text_result = ""
                
                if crop_img.size > 0:
                    ocr_res = ocr_model.ocr(crop_img, cls=False)
                    # Safety net to avoid empty cell crash
                    text_result = ""
                    try:
                        if ocr_res and ocr_res[0]:
                            text_result = " ".join([line[1][0] for line in ocr_res[0] if line and len(line) > 1 and line[1]])
                            text_result = text_result.strip()
                    except Exception as e:
                        print(f"      OCR parsed format error: {e}")
                
                cells.append({
                    "bbox": [round(c, 2) for c in cell_box],
                    "row": row_idx,
                    "col": col_idx,
                    "rowspan": max(1, rowspan),
                    "colspan": max(1, colspan),
                    "text": text_result
                })
    return cells

# ============================================================
# CELL 5: Run inference and build the submission JSON
# ============================================================
import glob
import json
import cv2
import gc
import torch
import traceback
from ultralytics import YOLO
from paddleocr import PaddleOCR

print(f"Loading model from: {weights_path}")
model = YOLO(weights_path)

print("Loading PaddleOCR...")
# Note: In Colab GPU, use_gpu=True works out of the box
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=False)

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
image_paths = [
    p for p in glob.glob(os.path.join(images_dir, '*'))
    if os.path.splitext(p)[1].lower() in valid_exts
]
print(f"Found {len(image_paths)} images to process.\n")

submission_data = []

for idx, img_path in enumerate(image_paths):
    filename = os.path.basename(img_path)
    print(f"[{idx+1}/{len(image_paths)}] Processing {filename}...")
    
    entry = {"filename": filename, "tables": []}
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        results = model.predict(img_path, imgsz=1024, conf=0.15, iou=0.45, verbose=False)
        boxes = results[0].boxes

        tables = []
        all_rows = []
        all_cols = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]

            if cls_id == 0:
                tables.append({"bbox": bbox, "cells": []})
            elif cls_id == 1:
                all_rows.append(bbox)
            elif cls_id == 2:
                all_cols.append(bbox)

        for i, table in enumerate(tables):
            tx1, ty1, tx2, ty2 = table['bbox']
            table['table_id'] = i

            t_rows = [r for r in all_rows if tx1 <= (r[0]+r[2])/2 <= tx2 and ty1 <= (r[1]+r[3])/2 <= ty2]
            t_cols = [c for c in all_cols if tx1 <= (c[0]+c[2])/2 <= tx2 and ty1 <= (c[1]+c[3])/2 <= ty2]

            table['cells'] = extract_cells_and_ocr(table['bbox'], t_rows, t_cols, img, ocr)
            table['bbox'] = [round(v, 2) for v in table['bbox']]
            
            entry["tables"].append(table)
            
    except Exception as e:
        print(f"  -> ERROR processing {filename}: {e}")
        traceback.print_exc()
        
    finally:
        submission_data.append(entry)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(submission_data, f, indent=4)

print(f"Done! Submission saved to: {output_json}")
print(f"Total tables detected:   {sum(len(item['tables']) for item in submission_data)}")
print(f"Total cells formulated:  {sum(len(t['cells']) for item in submission_data for t in item['tables'])}")

# ============================================================
# CELL 6: (Optional) Print the complete JSON structure
# ============================================================
if submission_data:
    print("\nComplete Output Data:")
    print(json.dumps(submission_data, indent=4))
