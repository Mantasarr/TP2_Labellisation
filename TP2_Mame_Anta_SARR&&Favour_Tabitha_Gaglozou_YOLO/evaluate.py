# Importer les bibliothèques nécessaires
from ultralytics import YOLO
from pathlib import Path

# Définir le dossier du projet
project_dir = Path(__file__).resolve().parent

# Définir le dossier des résultats d'évaluation
output_dir = project_dir / "evaluation_results"

# Charger les meilleurs poids du modèle entraîné
model = YOLO(
    str(project_dir / "runs/detect/runs/fruits_detection/weights/best.pt")
)

# Évaluer le modèle sur l'ensemble de test
metrics = model.val(
    data=str(project_dir / "dataset_fruits/data.yaml"),
    split="test",
    imgsz=640,
    plots=True,
    project=str(output_dir),
    name="YOLO_Test",
    exist_ok=True
)

# Afficher les métriques principales
print("\n===== RÉSULTATS SUR LE TEST =====")
print(f"Précision : {metrics.box.mp:.4f}")
print(f"Rappel : {metrics.box.mr:.4f}")
print(f"mAP@50 : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")

precision = metrics.box.mp
rappel = metrics.box.mr

f1 = (
    2 * precision * rappel / (precision + rappel)
    if precision + rappel > 0 else 0
)

print(f"F1-score global (à partir des moyennes) : {f1:.4f}")