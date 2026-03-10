import os
import glob
import json
from ultralytics import YOLO

# ============================================================
# Helper: Extract cells from rows and columns
# ============================================================
def extract_cells_from_rows_cols(table_bbox, rows, columns):
    """
    Given a table bounding box, a list of row boxes, and a list of column boxes,
    this function intersects rows and columns to formulate cells.
    row box: [x1, y1, x2, y2]
    column box: [x1, y1, x2, y2]
    """
    cells = []
    
    if not rows and not columns:
        # Fallback: Treat entire table as 1 cell (1x1)
        rows = [table_bbox]
        columns = [table_bbox]
    elif not rows:
        # Found columns but no rows. Assume 1 row spanning table height.
        rows = [table_bbox]
    elif not columns:
        # Found rows but no columns. Assume 1 column spanning table width.
        columns = [table_bbox]
        
    # Sort rows by Y-coordinate (top to bottom)
    rows = sorted(rows, key=lambda r: (r[1] + r[3]) / 2)
    # Sort columns by X-coordinate (left to right)
    columns = sorted(columns, key=lambda c: (c[0] + c[2]) / 2)
    
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(columns):
            # The intersection of a row and a column gives the cell's bbox
            # We take X-coordinates from the column and Y-coordinates from the row
            rx1, ry1, rx2, ry2 = row
            cx1, cy1, cx2, cy2 = col
            
            # Formulate the cell box, clamped to the table boundaries just in case
            x1 = max(cx1, table_bbox[0])
            y1 = max(ry1, table_bbox[1])
            x2 = min(cx2, table_bbox[2])
            y2 = min(ry2, table_bbox[3])
            
            # Ensure valid box (sometimes models predict slightly misaligned rows/cols)
            if x2 > x1 and y2 > y1:
                cells.append({
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "row": row_idx,
                    "col": col_idx
                })
                
    return cells

def main():
    weights_path = r'runs\detect\yolo_documents\pilot_table_detection\weights\best.pt'
    images_dir = os.path.join('images', 'val') 
    output_json = 'submission_output.json'
    
    if not os.path.exists(weights_path):
        weights_path = os.path.join('runs', 'detect', 'pilot_table_detection', 'weights', 'best.pt')
        
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = [p for p in glob.glob(os.path.join(images_dir, '*')) if os.path.splitext(p)[1].lower() in valid_exts]
    print(f"Found {len(image_paths)} images to process for submission.")
    
    submission_data = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Run inference. Lower conf finds faint lines; lower IOU removes heavily overlapping duplicate lines.
        results = model.predict(img_path, conf=0.15, iou=0.45, verbose=False)
        boxes = results[0].boxes
        
        tables = []
        all_rows = []
        all_cols = []
        
        # Separate detected boxes
        # 0: table, 1: table_row, 2: table_column
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
                
        # Assign Rows and Cols to their bounding Table
        for i, table in enumerate(tables):
            tx1, ty1, tx2, ty2 = table['bbox']
            table['table_id'] = i
            
            table_rows = []
            table_cols = []
            
            # Check row inside table (use center point)
            for row in all_rows:
                cy = (row[1] + row[3]) / 2.0
                cx = (row[0] + row[2]) / 2.0
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    table_rows.append(row)
                    
            # Check col inside table (use center point)
            for col in all_cols:
                cy = (col[1] + col[3]) / 2.0
                cx = (col[0] + col[2]) / 2.0
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    table_cols.append(col)
                    
            # Generate cells by intersection
            table['cells'] = extract_cells_from_rows_cols(table['bbox'], table_rows, table_cols)
            # Round table bbox as well
            table['bbox'] = [round(v, 2) for v in table['bbox']]
            
        submission_data.append({
            "filename": filename,
            "tables": tables
        })
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, indent=4)
        
    print(f"\nProcessing complete! Formatted JSON saved to {output_json}")

if __name__ == '__main__':
    main()
