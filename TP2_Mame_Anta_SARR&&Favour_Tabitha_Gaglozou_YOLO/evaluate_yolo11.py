from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

BEST_MODEL = (
    PROJECT_DIR
    / "runs_comparaison/YOLO11_fruits/weights/best.pt"
)

model = YOLO(str(BEST_MODEL))

metrics = model.val(
    data=str(PROJECT_DIR / "dataset_fruits/data.yaml"),
    split="test",
    imgsz=640,
    plots=True,
    project=str(PROJECT_DIR / "evaluation_results"),
    name="YOLO11_Test",
    exist_ok=True
)

precision = metrics.box.mp
recall = metrics.box.mr
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\nRÉSULTATS YOLO11")
print(f"Précision : {precision:.4f}")
print(f"Rappel : {recall:.4f}")
print(f"F1 (calculé sur les moyennes) : {f1:.4f}")
print(f"mAP@50 : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")