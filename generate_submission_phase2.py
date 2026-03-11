import os
import glob
import json
import cv2
import gc
import torch
import traceback
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_DIR = r'C:\Users\midob\Downloads\Hackez\ocr_images'
OUTPUT_JSON = 'final_submission.json'
YOLO_WEIGHTS = r'runs\detect\yolo_documents\pilot_table_detection\weights\best.pt'
CONF_THRESH = 0.15
IOU_THRESH = 0.45
IMGSZ = 1024  # High resolution for dense financial documents

def extract_cells_and_ocr(table_bbox, rows, columns, img, ocr_model):
    cells = []
    
    # --- FALLBACK LOGIC ---
    if not rows and not columns:
        rows = [table_bbox]
        columns = [table_bbox]
    elif not rows:
        rows = [table_bbox]
    elif not columns:
        columns = [table_bbox]
        
    # Sort for row/col index mapping
    rows = sorted(rows, key=lambda r: (r[1] + r[3]) / 2)
    columns = sorted(columns, key=lambda c: (c[0] + c[2]) / 2)
    
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(columns):
            cx1, cy1, cx2, cy2 = col
            rx1, ry1, rx2, ry2 = row
            
            # Intersection defining the cell
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
                    # If this cell bounds overlap significantly with the row's height
                    if r_overlap / max(1e-5, r[3] - r[1]) > 0.5: rowspan += 1
                for c in columns:
                    c_overlap = max(0, min(x2, c[2]) - max(x1, c[0]))
                    # If this cell bounds overlap significantly with the col's width
                    if c_overlap / max(1e-5, c[2] - c[0]) > 0.5: colspan += 1
                
                # --- PADDLE OCR TEXT EXTRACTION ---
                # Crop carefully so we don't go out of image bounds
                img_h, img_w = img.shape[:2]
                crop_y1 = max(0, int(y1))
                crop_y2 = min(img_h, int(y2))
                crop_x1 = max(0, int(x1))
                crop_x2 = min(img_w, int(x2))
                
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                text_result = ""
                
                # Only OCR if the crop is valid geometry and has reasonable dimensions
                if crop_img.shape[0] >= 5 and crop_img.shape[1] >= 5:
                    ocr_res = ocr_model.ocr(crop_img, cls=False)
                    # PaddleOCR returns nested lists. Extract text carefully.
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

def main():
    print(f"Loading YOLO Model on CUDA... ")
    model = YOLO(YOLO_WEIGHTS)
    
    print("Loading PaddleOCR...")
    # use_angle_cls=False speeds up extraction for upright tables
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [p for p in glob.glob(os.path.join(INPUT_DIR, '*')) if os.path.splitext(p)[1].lower() in valid_exts]
    
    print(f"Found {len(image_paths)} images in {INPUT_DIR}.")
    if len(image_paths) == 0:
        print("WARNING: No images found. Check the path!")
    
    submission_data = []
    processed_files = set()
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                submission_data = json.load(f)
            processed_files = {entry['filename'] for entry in submission_data}
            print(f"Resuming progress: {len(processed_files)} images already processed.")
        except Exception as e:
            print(f"Could not load previous JSON, starting fresh. {e}")
            submission_data = []
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        if filename in processed_files:
            # Skip already processed images
            continue
            
        print(f"[{idx+1}/{len(image_paths)}] Processing {filename}...")
        
        entry = {
            "filename": filename,
            "tables": []
        }
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            # 1. Run YOLO (Dynamic imgsz=1024, no pre-resizing required!)
            device_str = 0 if torch.cuda.is_available() else 'cpu'
            results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESH, iou=IOU_THRESH, device=device_str, verbose=False)
            boxes = results[0].boxes
            
            tables, all_rows, all_cols = [], [], []
            
            # Sort raw YOLO predictions by class
            for box in boxes:
                cls_id = int(box.cls[0].item())
                bbox = box.xyxy[0].tolist()
                if cls_id == 0:
                    tables.append({"bbox": bbox, "cells": []})
                elif cls_id == 1:
                    all_rows.append(bbox)
                elif cls_id == 2:
                    all_cols.append(bbox)
            
            # 2. Build Structural Cells & OCR
            for i, table in enumerate(tables):
                tx1, ty1, tx2, ty2 = table['bbox']
                table['table_id'] = i
                
                # Filter rows/cols to only the ones within THIS table's boundaries
                t_rows = [r for r in all_rows if tx1 <= (r[0]+r[2])/2 <= tx2 and ty1 <= (r[1]+r[3])/2 <= ty2]
                t_cols = [c for c in all_cols if tx1 <= (c[0]+c[2])/2 <= tx2 and ty1 <= (c[1]+c[3])/2 <= ty2]
                
                # Run the math to formulate cells, span logic, and crop out the image for PaddleOCR
                table['cells'] = extract_cells_and_ocr(table['bbox'], t_rows, t_cols, img, ocr)
                table['bbox'] = [round(v, 2) for v in table['bbox']]
                
                entry["tables"].append(table)
                
        except Exception as e:
            # Skip the image gracefully if an error occurs but keep the JSON intact
            print(f"  -> ERROR processing {filename}: {e}")
            traceback.print_exc()
            
        finally:
            submission_data.append(entry)
            
            # Save every 10 images to avoid data loss
            if (idx + 1) % 10 == 0:
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(submission_data, f, indent=4)
                print(f"  [Checkpoint: Saved intermediate progress to {OUTPUT_JSON} after {idx+1} images]")
            
            # --- CRITICAL: MEMORY MANAGEMENT ---
            # Clears internal CUDA cache across the 500 loop gracefully
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, indent=4)
        
    print(f"\nProcessing Complete! Ready for Machathon Sub-Phase 2.1.")
    print(f"Saved cleanly to: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
