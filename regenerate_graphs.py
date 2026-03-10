from ultralytics import YOLO
import os

def main():
    weights_path = r'runs\detect\yolo_documents\pilot_table_detection\weights\best.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}")
        return

    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)

    print("\nRegenerating Validation Graphics (F1 Curve, PR Curve, Confusion Matrix)...")
    # This automatically evaluates the model on the val set and produces all the necessary charts and metrics
    # It will save the output in a newly generated folder (usually runs\detect\val).
    metrics = model.val(
        data='data.yaml',   # Use full dataset data.yaml
        plots=True,         # Force generating the visual plots
        device='cpu',       # Run on CPU strictly as before
        batch=16,
        imgsz=640
    )
    
    # Let's also run a prediction with save=True to get some visualized output images in a fresh train/predict folder
    print("\nRegenerating sample output detections...")
    # Select a few random validation images
    import glob
    import random
    val_images = glob.glob(os.path.join('images', 'val', '*.jpg'))
    if val_images:
        samples = random.sample(val_images, min(5, len(val_images)))
        model.predict(source=samples, save=True, device='cpu', conf=0.25)
        
    print("\nDone! Validation charts and graphs are regenerated and saved in 'runs\detect\val' (or similar).")
    print("Sample detection images are saved in 'runs\detect\predict'.")

if __name__ == '__main__':
    main()
