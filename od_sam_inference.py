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
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for the FastSAM model (default: 0.4)')
    parser.add_argument('--iou', type=float, default=0.9, help='IoU threshold for non-maximum suppression (default: 0.9)')
    parser.add_argument('--show_mask', action='store_true', help='Show resulting binary mask')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Realsense pipeline and configure color stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Load YOLOv8 model
    yolo_model = YOLO(args.yolo_weight)

    # Load FastSAM model
    fastsam_model = FastSAM(args.fastsam_weight)

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)

    # Create OpenCV windows
    cv2.namedWindow('FastSAM Inference', cv2.WINDOW_AUTOSIZE)
    if args.show_mask:
        cv2.namedWindow('Annotation Mask', cv2.WINDOW_AUTOSIZE)

    # Main loop for capturing frames and performing object detection
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert RealSense color frame to numpy array
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
                    # Create FastSAMPrompt instance
                    prompt_process = FastSAMPrompt(color_image, fastsam_results, device=DEVICE)

                    # Use all bounding boxes for FastSAM prompt
                    ann = prompt_process.box_prompt(bboxes=bounding_boxes)

                    # Display FastSAM results in the window
                    img_with_annotations = prompt_process.plot_to_result(annotations=ann)
                    cv2.imshow('FastSAM Inference', img_with_annotations)

                    if args.show_mask:
                        # Convert annotations to binary mask numpy array
                        ann_masks = np.array(ann).astype(np.uint8)

                        # Overlay the annotation masks
                        ann_mask_overlay = np.sum(ann_masks, axis=0)

                        # Normalize the overlay to be in the range [0, 1] and convert to uint8
                        ann_mask_overlay_normalized = (ann_mask_overlay / np.max(ann_mask_overlay) * 255).astype(np.uint8)

                        # Show the overlaid annotation masks in another window
                        cv2.imshow('Annotation Mask', ann_mask_overlay_normalized)

                else:
                    # If no FastSAM results, show the original frame in the FastSAM window
                    cv2.imshow('FastSAM Inference', color_image)

            # Check for keyboard "q" press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
