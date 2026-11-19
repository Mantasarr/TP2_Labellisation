from ultralytics import YOLO
from pathlib import Path

# Dossier du projet
PROJECT_DIR = Path(__file__).resolve().parent

# Configuration du dataset
DATA_YAML = PROJECT_DIR / "dataset_fruits" / "data.yaml"

# Charger YOLO11 nano pré-entraîné
model = YOLO("yolo11n.pt")

# Entraîner YOLO11
model.train(
    data=str(DATA_YAML),
    epochs=20,
    imgsz=640,
    batch=8,
    project=str(PROJECT_DIR / "runs_comparaison"),
    name="YOLO11_fruits",
    exist_ok=True,
    seed=42
)