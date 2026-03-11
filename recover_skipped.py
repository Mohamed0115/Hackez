import os
import json
import cv2
import traceback
from ultralytics import YOLO
from paddleocr import PaddleOCR

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
    SKIPPED_IMAGES = [
        'test_288.jpg', 'test_384.jpg', 'test_426.jpg', 'test_427.jpg', 
        'test_428.jpg', 'test_429.jpg', 'test_43.jpg', 'test_430.jpg', 
        'test_431.jpg', 'test_432.jpg', 'test_482.jpg', 'test_483.jpg', 
        'test_484.jpg', 'test_485.jpg', 'test_486.jpg'
    ]

    INPUT_DIR = r'C:\Users\midob\Downloads\Hackez\ocr_images'
    WEIGHTS = r'runs\detect\yolo_documents\pilot_table_detection\weights\best.pt'
    OUTPUT_JSON = 'recovered_data.json'
    
    # 1. Provide slightly looser thresholds to pick up difficult tables
    CONF_THRESH = 0.10
    IOU_THRESH = 0.45
    # Increase image resolution for predictions to capture granular detail on small text/grids
    IMGSZ = 1280 
    
    print("Loading YOLO Model...")
    model = YOLO(WEIGHTS)
    
    print("Loading PaddleOCR in CPU Safe Mode...")
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
    
    recovered_results = []
    
    for filename in SKIPPED_IMAGES:
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"\n🛠️ Recovering {filename}...")
        
        entry = {
            "filename": filename,
            "tables": []
        }
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            # Predict in CPU mode to avoid C++ crashing bugs
            results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESH, iou=IOU_THRESH, device='cpu', verbose=False)
            boxes = results[0].boxes
            
            tables, all_rows, all_cols = [], [], []
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                bbox = box.xyxy[0].tolist()
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
                
            print(f"✅ Extracted {len(tables)} tables from {filename}")
                
        except Exception as e:
            print(f"❌ Failed again on {filename}: {e}")
            traceback.print_exc()
        finally:
            recovered_results.append(entry)
            
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(recovered_results, f, indent=4)
        
    print(f"\n🎉 Finished processing skipped images! Saved effectively to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
