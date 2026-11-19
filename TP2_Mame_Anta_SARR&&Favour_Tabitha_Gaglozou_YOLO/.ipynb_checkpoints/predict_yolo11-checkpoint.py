from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

model = YOLO(
    str(PROJECT_DIR / "runs_comparaison/YOLO11_fruits/weights/best.pt")
)

results = model.predict(
    source=str(PROJECT_DIR / "dataset_fruits/images/test"),
    conf=0.25,
    save=True,
    project=str(PROJECT_DIR / "predictions"),
    name="YOLO11_Predictions",
    exist_ok=True
)