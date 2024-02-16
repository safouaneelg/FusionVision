import cv2
import numpy as np
import torch
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import pyrealsense2 as rs
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection with YOLO and FastSAM')
    parser.add_argument('--yolo_weight', type=str, required=True, help='Path to YOLO weights file')
    parser.add_argument('--fastsam_weight', type=str, required=True, help='Path to FastSAM weights file (e.g., FastSAM-x.pt)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for YOLO detection (default: 0.7)')
    parser.add_argument('--bbox_color', type=str, default="red", help='Bounding box color (default: "red")')
    parser.add_argument('--font_scale', type=float, default=0.5, help='Font scale for displaying text (default: 0.5)')
    parser.add_argument('--font_thickness', type=int, default=1, help='Font thickness for displaying text (default: 1)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for the FastSAM model (default: 0.4)')
    parser.add_argument('--iou', type=float, default=0.9, help='IoU threshold for non-maximum suppression (default: 0.9)')
    parser.add_argument('--show_mask', action='store_true', help='Show resulting binary mask')
    return parser.parse_args()

def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Load YOLOv8 and FastSAM model
    yolo_model = YOLO(args.yolo_weight)
    fastsam_model = FastSAM(args.fastsam_weight)

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)

    cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)
    cv2.namedWindow('FastSAM Inference', cv2.WINDOW_NORMAL)

    if args.show_mask:
        cv2.namedWindow('Annotation Mask', cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Perform YOLO inference using the defined function
            detections, predicted_boxes = perform_yolo_inference(color_image, yolo_model, confidence_threshold=args.confidence_threshold)

            if len(predicted_boxes) > 0:
                # Extract all bounding boxes from YOLO predictions
                bounding_boxes = [list(map(int, box[:4])) for box in predicted_boxes]

                # Run FastSAM on the color image
                fastsam_results = fastsam_model(color_image, device=DEVICE, retina_masks=True, imgsz=640, conf=args.conf, iou=args.iou)

                # Check if there are FastSAM results
                if fastsam_results:
                    prompt_process = FastSAMPrompt(color_image, fastsam_results, device=DEVICE) # FastSAMPrompt instance
                    ann = prompt_process.box_prompt(bboxes=bounding_boxes) # Use all bounding boxes for FastSAM prompt

                    img_with_annotations = prompt_process.plot_to_result(annotations=ann)

                    cv2.imshow('FastSAM Inference', img_with_annotations)

                    if args.show_mask: # Convert annotations to binary mask numpy array
                        ann_masks = np.array(ann).astype(np.uint8)
                        ann_mask_overlay = np.sum(ann_masks, axis=0)
                        # Normalizing the overlay in the range [0, 1] and convert to uint8
                        ann_mask_overlay_normalized = (ann_mask_overlay / np.max(ann_mask_overlay) * 255).astype(np.uint8)

                        cv2.imshow('Annotation Mask', ann_mask_overlay_normalized)

                else:
                    # If no FastSAM results, show the original frame in the FastSAM window
                    cv2.imshow('FastSAM Inference', color_image)

            # Draw bounding boxes and display detection information on the image
            for detection in detections:
                x1, y1, x2, y2 = detection['bounding_box']
                confidence = detection['confidence']
                class_name = detection['class_name']

                cv2.rectangle(color_image, (x1, y1), (x2, y2), get_color(args.bbox_color), 3)

                # Display confidence and class name
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = args.font_scale
                color = (255, 255, 255)
                thickness = args.font_thickness

                cv2.putText(color_image, f"{class_name}: {confidence}", org, font, font_scale, color, thickness)

            cv2.imshow('YOLO Inference', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
