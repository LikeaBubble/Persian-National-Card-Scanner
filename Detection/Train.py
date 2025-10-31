from ultralytics import YOLO

yaml_path = '../Data/data.yaml'

model = YOLO('yolov8n.pt')

res = model.train(
    data = yaml_path,
    epochs=50,
    imgsz=1024,
    batch=8,
    lr0=0.01,
    lrf=0.001
)

