import os
import torchvision
from dotenv import load_dotenv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import torch
from .AbstractStrategyLayout import AbstractStrategyLayout

load_dotenv()


class StrategyFRCNN(AbstractStrategyLayout):
    def __init__(self, image, device=None, log=False):
        super().__init__(image, log)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") # Choose device: GPU if available
        self.layout_model_path = os.path.join(
            os.getenv('FASTERCRNN_LAYOUT'),
            "model_epoch_7.pth"
        )
        self.model = self.load_model() # Load the trained model
        self.label_map = {   # Dictionary to map label numbers
            1: "Caption",
            2: "Footnote",
            3: "Formula",
            4: "List-item",
            5: "Page-footer",
            6: "Page-header",
            7: "Picture",
            8: "Section-header",
            9: "Table",
            10: "Text",
            11: "Title"
        }

    def load_model(self):
        # Load a standard Faster R-CNN model (with a ResNet50 backbone)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features # number of features from the classifier head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=12) # Replace original names 12 = 11 + background
        model.load_state_dict(torch.load(self.layout_model_path, map_location=self.device))# Load trained model
        model.to(self.device)# Move the model to device
        model.eval()# evaluation mode
        if self.log:
            print(f"## Model loaded from: {self.layout_model_path}")
        return model

    def execute(self): # layout detection on input image
        image = self.load_image(self.image)        # Load input image
        # Convert image to tensor format, add batch dimension
        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        # inference without gradients (faster, use less memory)
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        # Extract bounding boxes, class labels and scores
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        # Keep predictions scored >= 0.5
        selected = scores >= 0.5
        boxes = boxes[selected]
        labels = labels[selected]
        scores = scores[selected]

        results = []
        for box, label, score in zip(boxes, labels, scores):
            result = {
                "box": [round(float(x), 2) for x in box],
                "label": int(label),
                "label_name": self.label_map.get(int(label), f"Class {label}"),
                "score": round(float(score), 4)
            }
            results.append(result)

        if self.log:
            print("###  Results saved to layout_results.json")
            print(f"###  Detected layout elements: {len(results)}")

        image_with_boxes = draw_boxes_on_image(image, results)
        return image_with_boxes, results

    def load_image(self, image_input):
        if isinstance(image_input, str):
            # If input is string, open the file
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # If Numpy array (-> OpenCV) convert to PIL
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError("Invalid image input")


def draw_boxes_on_image(image, detections, color=(255, 0, 0), thickness=8):
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

    for det in detections:# Draw each detection
        box = det["box"]
        label_name = det.get("label_name", f"Class {det['label']}")
        score = det["score"]

        x1, y1, x2, y2 = map(int, box) # Convert box values to integer
        cv.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        text = f"{label_name}: {score:.2f}"
        cv.putText(output, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 2, color, 5)   # text above the box

    return output
