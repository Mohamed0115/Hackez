import os
import random
from ultralytics import YOLO

def main():
    print("Setting up Pilot Dataset (500 train, 100 val)...")
    
    # Get list of all images
    train_images_dir = 'images/train'
    val_images_dir = 'images/val'
    
    # Gather all image files
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_train = [os.path.abspath(os.path.join(train_images_dir, f)) 
                 for f in os.listdir(train_images_dir) 
                 if f.lower().endswith(valid_exts)]
    all_val = [os.path.abspath(os.path.join(val_images_dir, f)) 
               for f in os.listdir(val_images_dir) 
               if f.lower().endswith(valid_exts)]
               
    if len(all_train) == 0 or len(all_val) == 0:
        print("Error: No images found in training or validation directories.")
        return
    
    # Randomly select subset
    random.seed(42)
    pilot_train = random.sample(all_train, min(500, len(all_train)))
    pilot_val = random.sample(all_val, min(100, len(all_val)))
    
    # Write paths to txt files (YOLO supports reading paths from txt files instead of directories)
    train_txt_path = os.path.abspath('train_pilot.txt')
    val_txt_path = os.path.abspath('val_pilot.txt')
    
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pilot_train) + '\n')
        
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pilot_val) + '\n')
        
    # Write pilot_data.yaml referencing the txt files
    yaml_content = f"""train: {train_txt_path.replace('\\', '/')}
val: {val_txt_path.replace('\\', '/')}

names:
  0: table
  1: table_row
  2: table_column
"""
    pilot_yaml_path = 'pilot_data.yaml'
    with open(pilot_yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    print(f"Pilot dataset created: {len(pilot_train)} train, {len(pilot_val)} val")

    # Load a pre-trained lightweight YOLOv8 model
    print("\nLoading YOLOv8 nano model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')

    print("\nStarting YOLOv8 PILOT fine-tuning on CPU...")
    
    results = model.train(
        data=pilot_yaml_path,      # Use our new pilot dataset mapper
        epochs=30,                 # Reduced to 30 for pilot run
        patience=5,                # Early stop faster if no improvement
        batch=16,                  
        imgsz=640,                 
        save=True,                 # Save best model
        project='yolo_documents',  
        name='pilot_table_detection', 
        
        # --- Hardware & Efficiency Fixes ---
        optimizer='AdamW',         # Much more efficient for CPU than Muon
        device='cpu',              # Force CPU
        workers=0,                 # Prevent multiprocessing errors on Windows
        
        # --- Data Augmentation suitable for Document Layouts ---
        degrees=2.0,               
        scale=0.2,                 
        shear=0.0,                 
        perspective=0.0,           
        mosaic=1.0,                
        mixup=0.1,                 
        copy_paste=0.0,            
        fliplr=0.0,                # Disable horizontal flip to preserve layout structure
        flipud=0.0,                # Disable vertical flip
        hsv_h=0.015,               
        hsv_s=0.4,                 
        hsv_v=0.4,                 
    )

    print("\n" + "="*50)
    print("TRAINING FINISHED")
    print("="*50)
    
    # Ultralytics results object has paths to the saved models
    print(f"Results natively saved to: {results.save_dir}")
    print(f"Best model weights natively saved at: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    
    # Evaluate the best model once training concludes
    print("\nEvaluating best model on pilot validation split...")
    metrics = model.val() # Automatically loads 'best.pt' for evaluation
    
    print("\n=== FINAL VALIDATION METRICS ===")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@75:    {metrics.box.map75:.4f}")

if __name__ == '__main__':
    main()
