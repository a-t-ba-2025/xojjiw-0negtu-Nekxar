import cv2 as cv
import torch
from PIL import Image, ImageDraw
import numpy as np
from .AbstractStrategyLayout import AbstractStrategyLayout
from .StrategyDETR import StrategyDETR
from .StrategyFRCNN import StrategyFRCNN


class StrategyHybridFRCNN_DETR(AbstractStrategyLayout):
    def __init__(self, image, device=None, log=False):
        super().__init__(image, log)
        self.device = device
        self.detr_strategy = StrategyDETR(image=image, device=device, log=log)
        self.frcnn_strategy = StrategyFRCNN(image=image, device=device, log=log)

    def execute(self):
        _, detr_boxes = self.detr_strategy.execute()
        _, frcnn_boxes = self.frcnn_strategy.execute()

        # detr or frcnn
        for box in detr_boxes:
            print(f"DETR: {box}")
            box["source"] = "DETR"
        for box in frcnn_boxes:
            print(f"FRCNN: {box}")
            box["source"] = "FRCNN"

        all_boxes = detr_boxes + frcnn_boxes
        image_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        image_with_boxes = draw_boxes_on_image(image_rgb, all_boxes)

        return np.array(image_with_boxes), all_boxes


def draw_boxes_on_image(image, detections, thickness=2):
    # If image is PIL format, convert  to Numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If image is PyTorch tensor, convert to NumPy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # If the image is float convert to uint8
    if isinstance(image, np.ndarray) and image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # If image is grayscale, convert to color (BGR)
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    output = image.copy()

    for det in detections:  # Draw each detection
        box = det["box"]
        label_name = det.get("label_name", f"Class {det['label']}")
        score = det["score"]
        source = det["source"]
        color = (0, 0, 255) if source == "FRCNN" else (255, 0, 0)

        x1, y1, x2, y2 = map(int, box)  # Convert box values to integer
        cv.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        text = f"{label_name}: {score:.2f}, Source: {source}"
        cv.putText(output, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # text above the box

    return output

