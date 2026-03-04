from ultralytics import YOLO

def train_model():
    print("🚗 Starting BFMC Traffic Sign Training on Apple Silicon (MPS)...")
    
    # Initialize the base Nano model (fastest and lightest for Jetson/Pi)
    model = YOLO("yolov8n.pt")
    
    # Train the model
    # We use 'mps' (Metal Performance Shaders) device to utilize your M2 GPU properly!
    results = model.train(
        data="Bosch Traffic Signs YOLOv8/data.yaml",      # The config file from the Roboflow zip
        epochs=100,            # 100 epochs is a good baseline for YOLOv8 and 641 images
        imgsz=640,             # Keep the image size at 640x640 (standard)
        device="mps",          # 🔥 Force Apple Silicon Hardware Acceleration
        batch=16,              # Safe amount for 8GB RAM, you can lower to 8 if it crashes
        project="bfmc_models", # Where to save the output folder
        name="sign_detector"   # The output folder name
    )

    print("\n✅ Training Complete!")
    print("Your new offline model weights are located at:")
    print("./bfmc_models/sign_detector/weights/best.pt")

if __name__ == "__main__":
    train_model()
