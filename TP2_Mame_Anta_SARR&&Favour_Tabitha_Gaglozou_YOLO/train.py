# Importer YOLO depuis la bibliothèque Ultralytics
from ultralytics import YOLO

# Charger un modèle YOLO26n pré-entraîné
model = YOLO("yolo26n.pt")

# Lancer le fine-tuning sur notre dataset
model.train(
    data="dataset_fruits/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    project="runs",
    name="fruits_detection"
)