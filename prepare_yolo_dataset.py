import os
import json
import shutil
import random
import glob

def main():
    all_images_dir = 'all_images'
    
    # Check for annotations folder
    if os.path.isdir('annotations'):
        annotations_dir = 'annotations'
    elif os.path.isdir('annotation'):
        annotations_dir = 'annotation'
    else:
        print("Error: Could not find annotations folder.")
        return

    json_files = glob.glob(os.path.join(annotations_dir, '*.json'))
    if not json_files:
        print(f"Error: No JSON files found in {annotations_dir}.")
        return

    # To group all annotations for the same image across different JSONs
    # Mapping: image_filename -> { 'width': w, 'height': h, 'bboxes': [(class_id, xc, yc, w, h), ...] }
    images_data = {}

    for jf_path in json_files:
        print(f"Parsing {jf_path}...")
        with open(jf_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 1. Map category_id to YOLO class (0: table, 1: table row, 2: table column)
        cat_to_yolo = {}
        if 'categories' in data:
            for cat in data['categories']:
                name = cat.get('name', '').lower()
                if name == 'table':
                    cat_to_yolo[cat['id']] = 0
                elif name == 'table row':
                    cat_to_yolo[cat['id']] = 1
                elif name == 'table column':
                    cat_to_yolo[cat['id']] = 2
        
        # 2. Map image_id to filename, width, height
        images_mapping = {}
        if 'images' in data:
            for img in data['images']:
                file_name = os.path.basename(img['file_name'])
                images_mapping[img['id']] = {
                    'file_name': file_name,
                    'width': img.get('width'),
                    'height': img.get('height')
                }

        # 3. Process annotations
        if 'annotations' in data:
            for ann in data['annotations']:
                cat_id = ann.get('category_id')
                if cat_id not in cat_to_yolo:
                    continue  # Ignore other categories (like spanning cells)
                    
                yolo_class = cat_to_yolo[cat_id]
                img_id = ann.get('image_id')
                
                if img_id not in images_mapping:
                    continue
                    
                img_info = images_mapping[img_id]
                file_name = img_info['file_name']
                img_w = img_info['width']
                img_h = img_info['height']
                
                if not img_w or not img_h:
                    continue
                
                # COCO format: [x_min, y_min, width, height]
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                    
                x_min, y_min, w, h = bbox
                
                # YOLO format: [x_center, y_center, width, height] normalized
                x_center = (x_min + w / 2.0) / img_w
                y_center = (y_min + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Clamp values
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                if file_name not in images_data:
                    images_data[file_name] = {
                        'width': img_w,
                        'height': img_h,
                        'bboxes': []
                    }
                    
                images_data[file_name]['bboxes'].append((yolo_class, x_center, y_center, w_norm, h_norm))


    # 4. Filter only images that actually exist in the images folder
    valid_images = []
    for file_name in images_data.keys():
        src_img1 = os.path.join(all_images_dir, file_name)
        src_img2 = os.path.join('images', 'train', file_name)
        src_img3 = os.path.join('images', 'val', file_name)
        if os.path.exists(src_img1):
            valid_images.append((file_name, src_img1))
        elif os.path.exists(src_img2):
            valid_images.append((file_name, src_img2))
        elif os.path.exists(src_img3):
            valid_images.append((file_name, src_img3))
        else:
            print(f"Warning: {file_name} found in JSON, but missing from {all_images_dir}/images.")
            
    if not valid_images:
        print("Error: No images found from the JSONs!")
        return

    # Sort to ensure reproducibility before shuffling
    valid_images.sort(key=lambda x: x[0])
    
    # 5. Split train/val (80% / 20%)
    random.seed(42)
    random.shuffle(valid_images)
    num_train = int(len(valid_images) * 0.8)
    train_files = valid_images[:num_train]
    val_files = valid_images[num_train:]

    # 6. Make directories
    dirs_to_create = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    # 7. Move files and generate YOLO txt
    def process_split(split_files, split_name):
        moved_count = 0
        for f_name, src_img in split_files:
            dst_img = os.path.join('images', split_name, f_name)
            
            # Move the image
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img) # Use copy instead of move if already done
                moved_count += 1
            elif src_img != dst_img and os.path.exists(src_img):
                 shutil.move(src_img, dst_img)
                 moved_count += 1

            
            # Create label file
            txt_name = os.path.splitext(f_name)[0] + '.txt'
            dst_txt = os.path.join('labels', split_name, txt_name)
            
            with open(dst_txt, 'w', encoding='utf-8') as out:
                for bbox in images_data[f_name]['bboxes']:
                    # format: class_id x_center y_center width height
                    res_str = f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
                    out.write(res_str)
                    
        return moved_count

    print(f"\nProcessing {len(train_files)} files for training...")
    train_moved = process_split(train_files, 'train')
    
    print(f"Processing {len(val_files)} files for validation...")
    val_moved = process_split(val_files, 'val')

    # 8. Generate data.yaml
    yaml_content = f"""train: images/train
val: images/val

names:
  0: table
  1: table_row
  2: table_column
"""
    with open('data.yaml', 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    print("\n--- Summary ---")
    print(f"Total annotations matched: {sum(len(info['bboxes']) for info in images_data.values())}")
    print("Labels successfully generated for table (0), table_row (1), and table_column (2).")
    print("Dataset preparation complete! 'data.yaml' has been updated.")

if __name__ == '__main__':
    main()
