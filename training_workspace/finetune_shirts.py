from ultralytics import YOLO
import shutil
import os

def finetune_with_shirts():
    print("👕 Fine-tuning BFMC model with shirt negative mining data...")
    
    # Load the already-trained 100-epoch model
    model = YOLO("../src/perception/sign_recognition/bfmc_best.pt")
    
    # Fine-tune for 10 more epochs on the updated dataset (now includes shirts)
    results = model.train(
        data="Bosch Traffic Signs YOLOv8/data.yaml",
        epochs=10,
        imgsz=640,
        device="mps",
        batch=16,
        project="bfmc_models",
        name="sign_detector_shirts"
    )

    # Copy the output weights to the sign_recognition folder
    best_src = "../runs/detect/bfmc_models/sign_detector_shirts/weights/best.pt"
    last_src = "../runs/detect/bfmc_models/sign_detector_shirts/weights/last.pt"
    best_dst = "../src/perception/sign_recognition/bfmc_best_shirts.pt"
    last_dst = "../src/perception/sign_recognition/bfmc_last_shirts.pt"

    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"\n✅ Copied best weights to: {best_dst}")
    if os.path.exists(last_src):
        shutil.copy(last_src, last_dst)
        print(f"✅ Copied last weights to: {last_dst}")

    print("\n✅ Fine-tuning Complete!")
    print("Your shirt-aware model weights are at:")
    print("  src/perception/sign_recognition/bfmc_best_shirts.pt")
    print("  src/perception/sign_recognition/bfmc_last_shirts.pt")

if __name__ == "__main__":
    finetune_with_shirts()
