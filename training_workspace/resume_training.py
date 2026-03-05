from ultralytics import YOLO

def resume_training():
    print("🔄 Resuming BFMC Traffic Sign Training on Apple Silicon (MPS)...")
    
    # Load the partially trained model
    model = YOLO("../runs/detect/bfmc_models/sign_detector/weights/last.pt")
    
    # Resume the training for 10 more epochs (target 90 total)
    results = model.train(resume=True, epochs=100)

    import shutil
    import os
    
    # Save the best weights to the requested location automatically
    best_weights_path = "../runs/detect/bfmc_models/sign_detector/weights/best.pt"
    target_path = "../src/perception/sign_recognition/bfmc_best.pt"
    
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, target_path)
        print(f"\n✅ Automatically copied finalized model to: {target_path}")


    print("\n✅ Training Complete!")
    print("Your new offline model weights are located at:")
    print("../runs/detect/bfmc_models/sign_detector/weights/best.pt")

if __name__ == "__main__":
    resume_training()
