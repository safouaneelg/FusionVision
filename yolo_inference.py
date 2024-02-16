import cv2
import numpy as np
import torch
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import pyrealsense2 as rs
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Inference with Realsense camera.')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights file.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for YOLO inference (default: 0.7)')
    parser.add_argument('--bbox_color', type=str, default="red", help='Bounding box color (default: "red")')
    parser.add_argument('--font_scale', type=float, default=0.5, help='Font scale for displaying text (default: 0.5)')
    parser.add_argument('--font_thickness', type=int, default=1, help='Font thickness for displaying text (default: 1)')
    return parser.parse_args()

def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Load YOLOv8 model
    yolo_model = YOLO(args.weights)

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)

    cv2.namedWindow('YOLO Inference', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            detections, _ = perform_yolo_inference(color_image, yolo_model, confidence_threshold=args.confidence_threshold)

            for detection in detections:
                x1, y1, x2, y2 = detection['bounding_box']
                confidence = detection['confidence']
                class_name = detection['class_name']

                color = get_color(args.bbox_color)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)

                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_image, f"{class_name}: {confidence}", org, font, args.font_scale, color, args.font_thickness)

            cv2.imshow('YOLO Inference', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
