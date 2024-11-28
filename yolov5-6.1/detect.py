import cv2
import os
from pathlib import Path
import torch
from yolov11 import YOLO  # 假设 YOLOv11 提供一个与 YOLOv5 类似的接口

def detect_and_draw_yolov11(weights, source, output_dir, conf_thres=0.25, iou_thres=0.45, device='cuda:0'):
    """
    Perform object detection using YOLOv11 model and draw bounding boxes on the images.

    Args:
        weights (str): Path to the YOLO model weights.
        source (str): Directory containing images or path to a single image.
        output_dir (str): Directory to save the output images.
        conf_thres (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for non-max suppression.
        device (str): Device to run inference on.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 假设 YOLOv11 模型加载方式与 YOLOv5 类似
    model = YOLO(weights)
    model.to(device)

    # 输入验证
    if not os.path.exists(source):
        print(f"Source {source} does not exist.")
        return

    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        files = [source]

    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to read image {file_path}")
            continue

        # YOLOv11 推理
        results = model.predict(source=img, save=False, device=device, conf=conf_thres, iou=iou_thres)

        # 注释结果在图片上
        annotator = cv2.dnn.DetectionModel(results)  # 假设 YOLOv11 提供此方式进行注释
        annotated_img = annotator.result()  # 获取注释后的图片

        save_path = os.path.join(output_dir, Path(file_path).name)
        cv2.imwrite(save_path, annotated_img)
        print(f"Processed {file_path} and saved to {save_path}")

if __name__ == "__main__":
    # 设定路径
    weights_path = 'path_to_your_yolov11_weights'  # 替换为 YOLOv11 权重文件路径
    source_dir = 'path_to_your_input_images'  # 输入图像目录
    output_dir = 'path_to_your_output_directory'  # 输出目录

    detect_and_draw_yolov11(weights=weights_path, source=source_dir, output_dir=output_dir)
