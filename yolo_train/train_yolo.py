import argparse
from ultralytics import YOLO

def train_yolo(data_path, epochs=100):
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
    results = model.train(data=data_path, epochs=epochs, imgsz=640)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('--data', type=str, required=True, help='Path to the YAML file containing training data configuration.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training (default: 100)')

    args = parser.parse_args()
    train_yolo(data_path=args.data, epochs=args.epochs)