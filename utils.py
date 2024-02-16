# utils.py
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from ultralytics import YOLO
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch

should_save_image = False
close_captured_image_window = False

def get_color(color_name):
    color_dict = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'purple': (128, 0, 128),
        'orange': (0, 165, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'pink': (203, 192, 255),
        'teal': (128, 128, 0),
        'lime': (0, 255, 0),
        'brown': (42, 42, 165),
        'maroon': (0, 0, 128),
        'navy': (128, 0, 0),
        'olive': (0, 128, 128),
        'gray': (128, 128, 128),
        'silver': (192, 192, 192),
        'gold': (0, 215, 255),
        'turquoise': (208, 224, 64),
        'violet': (211, 0, 148),
        'indigo': (130, 0, 75),
        'lavender': (208, 184, 170),
        'peach': (255, 218, 185),
        'salmon': (114, 128, 250),
        'sky_blue': (235, 206, 135),
        'tan': (140, 180, 210),
        'dark_green': (0, 100, 0),
        'dark_red': (0, 0, 139),
        'dark_blue': (139, 0, 0),
    }

    return color_dict.get(color_name, (0, 0, 255))  # Default to red if color_name is not found

def check_camera_connection():
    """Check if camera is connected"""
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        context = rs.context() 
        devices = context.query_devices()
        for device in devices:
            print("Device Information:")
            print("Device Product Line: ", str(device.get_info(rs.camera_info.product_line)))
            print("Device Serial Number: ", str(device.get_info(rs.camera_info.serial_number)))
        pipeline.stop()
    except:
        print("Camera not connected!")
        exit()

def start_realtime_stream():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    global should_save_image, close_captured_image_window

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)

            key = cv2.waitKey(1)
            if key == ord('c'):
                capture_image(color_image)
            elif key == ord('s'):
                should_save_image = True

            if close_captured_image_window:
                if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) <= 0:
                    close_captured_image_window = False
                else:
                    cv2.destroyWindow('Captured Image')

            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def capture_image(image):
    cv2.namedWindow('Captured Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Captured Image', image)
    cv2.setWindowTitle('RealSense', 'Press \'s\' to save')

    global should_save_image, close_captured_image_window

    while True:
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            should_save_image = True
            close_captured_image_window = True
        elif key == 27:
            break

        if close_captured_image_window:
            if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) <= 0:
                close_captured_image_window = False
            else:
                cv2.destroyWindow('Captured Image')
                break

    if should_save_image:
        save_image(image)

def save_image(image):
    global should_save_image
    if should_save_image:
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        image_name = 'dataset/' + str(int(time.time() * 1000)) + '.png'
        cv2.imwrite(image_name, image)
        print('Image saved at:', image_name)
        should_save_image = False

def detect_and_visualize_yolo(input_data, yolo_model_path=None):
    if isinstance(input_data, str):  
        frame = cv2.imread(input_data)
    else:  
        frame = input_data

    if yolo_model_path is None:
        yolo_model_path = 'yolov8m.pt'
    model = YOLO(yolo_model_path)

    results = model.predict(source=frame, conf=0.50)
    predicted_boxes = results[0].boxes.xyxy.cpu().numpy()

    _, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    predicted_boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to NumPy on CPU

    for box in predicted_boxes:
        x, y, w, h = box[:4]
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()


def perform_yolo_inference(frame, model, confidence_threshold=0.6):
    results = model(frame, stream=True)
    detections = []
    predicted_boxes = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            if box.conf[0] >= confidence_threshold:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence and class name
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = model.names[cls]

                detections.append({
                    'bounding_box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_name': class_name
                })
                predicted_boxes.append([x1, y1, x2, y2])

    return detections, predicted_boxes

def run_realtime_object_detection():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    yolo_model = YOLO("yolo_train/runs/detect/train/weights/best.pt")

    fastsam_model = FastSAM('FastSAM-s.pt')

    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(DEVICE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            yolo_results = yolo_model.predict(source=color_image, conf=0.50)
            predicted_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

            if len(predicted_boxes) > 0:
                first_box = predicted_boxes[0]

                bounding_box = [int(i) for i in first_box[:4]]

                fastsam_results = fastsam_model(color_image, device=DEVICE, retina_masks=True, imgsz=640, conf=0.3, iou=0.7)

                prompt_process = FastSAMPrompt(color_image, fastsam_results, device=DEVICE)

                ann = prompt_process.box_prompt(bbox=bounding_box)

                img_with_annotations = prompt_process.plot_to_result(annotations=ann)

                cv2.imshow('RealSense + YOLO + FastSAM', cv2.addWeighted(color_image, 1, img_with_annotations, 0.5, 0))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    check_camera_connection()
    start_realtime_stream()