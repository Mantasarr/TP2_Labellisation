# Importer les bibliothèques nécessaires
from ultralytics import YOLO
from pathlib import Path

# Définir le dossier du projet
project_dir = Path(__file__).resolve().parent

# Charger le modèle entraîné
model = YOLO(
    str(project_dir / "runs/detect/runs/fruits_detection/weights/best.pt")
)

# Définir le dossier des prédictions
output_dir = project_dir / "predictions"

# Détecter les fruits dans les images de test
results = model.predict(
    source=str(project_dir / "dataset_fruits/images/test"),
    conf=0.25,
    save=True,
    project=str(output_dir),
    name="YOLO_Predictions",
    exist_ok=True
)