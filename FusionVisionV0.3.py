import cv2
import numpy as np
import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import open3d as o3d
import pyrealsense2 as rs
from utils import perform_yolo_inference
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Vision Object Detection and 3D Reconstruction")
    
    parser.add_argument("--yolo_weight", type=str, required=True,
                        help="Path to the YOLO weights file (e.g yolo_train/runs/detect/train/weights/best.pt)")
    parser.add_argument("--fastsam_weight", type=str, choices=['FastSAM-x.pt', 'FastSAM-s.pt'], default='FastSAM-x.pt',
                        help="Choose the FastSAM autodownloadable weight files ('FastSAM-x.pt' or 'FastSAM-s.pt')")
    parser.add_argument("--show_yolo", action='store_true',
                        help="Show cv2 window with YOLO detection (default True)")
    parser.add_argument("--show_fastsam", action='store_true',
                        help="Show cv2 window with FastSAM detection (default True)")
    parser.add_argument("--show_mask", action='store_true',
                        help="Show a window with the estimated binary masks (default True)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Set the confidence threshold for YOLO detection (default: 0.7)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Set the confidence threshold for the FastSAM model (default: 0.4)")
    parser.add_argument("--iou", type=float, default=0.9,
                        help="Set the IoU threshold for non-maximum suppression (default: 0.9)")
    parser.add_argument("--show_3dbbox", action='store_true',
                        help="Show in open3D window the 3D bounding box (default: True)")
    
    return parser.parse_args()

def FusionVision(args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    yolo_model = YOLO(args.yolo_weight)

    fastsam_model_path = args.fastsam_weight
    fastsam_model = FastSAM(fastsam_model_path)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window("Point Cloud Viewer", width=640, height=480, visible=True)

    pcd = o3d.geometry.PointCloud()

    # Transformation matrix for flipping the point cloud upside down and left to right
    flip_matrix = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])

    if args.show_fastsam:
        cv2.namedWindow('FastSAM Inference', cv2.WINDOW_NORMAL)
    if args.show_mask:
        cv2.namedWindow('Annotation Mask', cv2.WINDOW_NORMAL)
    if args.show_mask:
        cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)

    # Create an align object
    align_to = rs.align(rs.stream.color)

    coordinates_list=[]

    try:
        while True:
            # Capture frames from RealSense camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            aligned_frames = align_to.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # Perform YOLO inference on the color image
            detections, predicted_boxes = perform_yolo_inference(color_image, yolo_model, confidence_threshold=args.confidence_threshold)

            # Create a list to store bounding box lines
            bounding_box_lines = []

            if len(predicted_boxes) > 0:
                bounding_boxes = [list(map(int, box[:4])) for box in predicted_boxes]

                fastsam_results = fastsam_model(color_image, device='cuda:0', retina_masks=True, imgsz=640, conf=args.conf,
                                                iou=args.iou)

                if fastsam_results:
                    prompt_process = FastSAMPrompt(color_image, fastsam_results, device='cuda:0')
                    ann = prompt_process.box_prompt(bboxes=bounding_boxes)
                    img_with_annotations = prompt_process.plot_to_result(annotations=ann)

                    # Display FastSAM results in another window
                    if args.show_fastsam:
                        cv2.imshow('FastSAM Inference', img_with_annotations)

                    ann_mask = np.array(ann).astype(np.uint8)
                    ann_mask_overlay = np.sum(ann_mask, axis=0)

                    # Normalize the overlay to be in the range [0, 1] and convert to uint8
                    ann_mask_overlay_normalized = (ann_mask_overlay / np.max(ann_mask_overlay) * 255).astype(np.uint8)

                    if args.show_mask:
                        cv2.imshow('Annotation Mask', ann_mask_overlay_normalized)

                    ann_mask_overlay_uint8 = ann_mask_overlay.astype(np.uint8)

                    # Erode the annotation mask (to avoid reconstructing in 3D some background)
                    eroded_ann_mask = cv2.erode(ann_mask_overlay_uint8, kernel=np.ones((20, 20), np.uint8), iterations=1)
                    isolated_depth = np.where((eroded_ann_mask > 0) & (depth_image < 1000), depth_image, np.nan)
                    non_nan_points = np.argwhere(~np.isnan(isolated_depth))
                    non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]

                    depth_scale = 1

                    pcd.points = o3d.utility.Vector3dVector(
                        np.column_stack([non_nan_points[:, 1], non_nan_points[:, 0], non_nan_depth_values * depth_scale])
                    )

                    pcd_outlier = pcd.voxel_down_sample(voxel_size=2)

                    denoised_pcd, _ = pcd_outlier.remove_statistical_outlier(nb_neighbors=300,
                                                                            std_ratio=2.0)

                    for box in bounding_boxes:
                        x_min, y_min, x_max, y_max = box

                        # Extract the region of interest from the denoised point cloud
                        roi_points = np.asarray(denoised_pcd.points)
                        roi_points = roi_points[(roi_points[:, 0] >= x_min) & (roi_points[:, 0] <= x_max) &
                                                (roi_points[:, 1] >= y_min) & (roi_points[:, 1] <= y_max)]

                        # Compute the bounding box dimensions based on the region of interest
                        bbox_lines = o3d.geometry.LineSet()
                        bbox_lines.points = o3d.utility.Vector3dVector([
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.min(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.max(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.min(roi_points[:, 2])],
                            [np.min(roi_points[:, 0]), np.max(roi_points[:, 1]), np.max(roi_points[:, 2])],
                        ])
                        bbox_lines.lines = o3d.utility.Vector2iVector([
                            [0, 1], [1, 2], [2, 3], [3, 0],
                            [4, 5], [5, 6], [6, 7], [7, 4],
                            [0, 7], [1, 6], [2, 5], [3, 4],
                            [8, 9], [9, 10], [10, 11], [11, 8],
                            [12, 13], [13, 14], [14, 15], [15, 12],
                            [8, 15], [9, 14], [10, 13], [11, 12],
                            [16, 17], [17, 18], [18, 19], [19, 16],
                            [20, 21], [21, 22], [22, 23], [23, 20],
                            [16, 23], [17, 22], [18, 21], [19, 20]
                        ])
                        bbox_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(bbox_lines.lines))])

                        bounding_box_lines.append(bbox_lines)

                    denoised_pcd.transform(flip_matrix)

                    visualizer.clear_geometries()
                    visualizer.add_geometry(denoised_pcd)

                    if args.show_3dbbox:
                        for bbox_lines in bounding_box_lines:
                            bbox_lines.transform(flip_matrix)
                            visualizer.add_geometry(bbox_lines)

                            center = np.mean(np.asarray(bbox_lines.points), axis=0)

                            coordinates_list.append(center)

                            # length of the coordinate axes
                            axis_length = 50

                            # Create coordinate frame mesh
                            coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=center)

                            visualizer.add_geometry(coordinate_system)

                    visualizer.poll_events()
                    visualizer.update_renderer()

                    csv_file_path = 'object_coordinates.csv'
                    # Write the coordinates to the CSV file
                    with open(csv_file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        # Write header if needed
                        writer.writerow(['X', 'Y', 'Z'])
                        # Write each set of coordinates
                        for coordinates in coordinates_list:
                            writer.writerow(coordinates)

            # if YOLO window = true
            if args.show_yolo:
                for detection in detections:
                    x1, y1, x2, y2 = detection['bounding_box']
                    confidence = detection['confidence']
                    class_name = detection['class_name']

                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0))

                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 255, 255)
                    thickness = 2

                    cv2.putText(color_image, f"{class_name}: {confidence}", org, font, font_scale, color, thickness)                
                    cv2.imshow('YOLO Inference', color_image)

            # Check for key press to exit the loop (press 'q' to quit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

    finally:
        # Stop streaming
        pipeline.stop()
        visualizer.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    FusionVision(args)
