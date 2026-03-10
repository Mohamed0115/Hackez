import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    # 1. Set paths for the best model
    weights_path = r'runs\detect\yolo_documents\pilot_table_detection\weights\best.pt'
    
    # Just in case the directory path was slightly differently named
    if not os.path.exists(weights_path):
        weights_path = os.path.join('runs', 'detect', 'pilot_table_detection', 'weights', 'best.pt')
        if not os.path.exists(weights_path):
            print(f"Error: Could not find model weights at {weights_path}")
            return
            
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    
    # 2. Pick 5 random images from the validation folder
    val_images_dir = os.path.join('images', 'val')
    if not os.path.isdir(val_images_dir):
        print(f"Error: Validation folder {val_images_dir} does not exist.")
        return
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_val_images = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir) 
                      if f.lower().endswith(valid_exts)]
                      
    if not all_val_images:
        print("Error: No images found in the validation directory.")
        return
        
    num_samples = min(5, len(all_val_images))
    # Seed without a fixed number to ensure different images every time this script is run
    random.seed()
    sample_images = random.sample(all_val_images, num_samples)
    
    # 3. Dedicated output directory for results
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    print(f"Running inference on {num_samples} images and displaying the results...\n")
    
    # Set up matplotlib figure (dynamically adjust columns based on samples)
    plt.figure(figsize=(20, 6))
    
    for i, img_path in enumerate(sample_images):
        print(f"Processing: {img_path}")
        
        # Run YOLO inference
        results = model.predict(img_path, conf=0.25) # conf=0.25 ignores very low confidence boxes
        
        # Since we pass one image path, it returns a list of length 1
        result = results[0]
        
        # The .plot() function generates an annotated numpy array (BGR format for cv2) with labels and boxes
        annotated_bgr = result.plot()
        
        # Save output for the presentation via cv2
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(save_path, annotated_bgr)
        print(f" -> Saved annotated image to: {save_path}")
        
        # Convert BGR to RGB for correct matplotlib color visualization
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Plot into grid (1 row, 'num_samples' columns)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(annotated_rgb)
        plt.title(f"Image {i+1}", fontsize=12)
        plt.axis('off')
        
    print(f"\nInference complete! 5 results saved in '{output_dir}'.")
    print("Close the Matplotlib window to end the script!")
    
    # Show the images on screen automatically
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
