from pathlib import Path
import shutil

from ultralytics import YOLO


def _choose_base_model(repo_root: Path) -> str:
    candidates = [
        repo_root / "src/perception/sign_recognition/bfmc_best_shirts.pt",
        repo_root / "src/perception/sign_recognition/bfmc_best.pt",
        repo_root / "src/perception/sign_recognition/best.pt",
        repo_root / "training_workspace/yolov8n.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "yolov8n.pt"


def _choose_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model():
    workspace_root = Path(__file__).resolve().parent
    repo_root = workspace_root.parent

    dataset_yaml = workspace_root / "Bosch Traffic Signs YOLOv8" / "data.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    base_model_path = _choose_base_model(repo_root)
    device = _choose_device()

    print("Starting BFMC sign + traffic-light fine-tuning")
    print(f"Base model: {base_model_path}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Device: {device}")

    model = YOLO(base_model_path)
    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        workers=0,
        cache=False,
        project=str(repo_root / "runs" / "detect" / "bfmc_models"),
        name="sign_detector",
        exist_ok=True,
    )

    best_src = repo_root / "runs" / "detect" / "bfmc_models" / "sign_detector" / "weights" / "best.pt"
    best_dst = repo_root / "src" / "perception" / "sign_recognition" / "bfmc_best_traffic_lights.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"Copied fine-tuned weights to: {best_dst}")
    else:
        print(f"Training finished but best.pt was not found at: {best_src}")


if __name__ == "__main__":
    train_model()
