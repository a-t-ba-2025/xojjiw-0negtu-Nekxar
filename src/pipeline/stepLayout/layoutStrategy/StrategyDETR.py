import os
import torch
from dotenv import load_dotenv
from transformers import DetrConfig
from PIL import Image
import numpy as np
import cv2 as cv
from transformers import DetrImageProcessor, DetrForObjectDetection
from .AbstractStrategyLayout import AbstractStrategyLayout

load_dotenv()


class StrategyDETR(AbstractStrategyLayout):
    def __init__(self, image, device=None, log=False):
        super().__init__(image, log)

        # Use GPU if available, otherwise use CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.layout_model_path = os.path.join(
            os.getenv('DETR_LAYOUT'),
            "detr_epoch_32.pth"
        )
        self.layout_model_processor_path = os.path.join(
            os.getenv('DETR_PROCESSOR')
        )

        # Load model and processor
        self.model, self.processor = self.load_model()

        # Map class IDs to  names (like DocLayNet)
        self.label_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title"
        }

    def load_model(self):
        processor_path = self.layout_model_processor_path
        processor = DetrImageProcessor.from_pretrained(processor_path)

        # Load model configuration with the number of labels (11)
        config = DetrConfig.from_pretrained("facebook/detr-resnet-50", num_labels=11)

        # Initialize DETR model and load trained weights
        model = DetrForObjectDetection(config)
        model.load_state_dict(torch.load(self.layout_model_path, map_location=self.device))

        model.to(self.device)  # Move model to GPU or CPU
        model.eval()  # evaluation mode

        if self.log:
            print(f"## Custom DETR model loaded from: {self.layout_model_path}")
            print(f"## Processor loaded from: {processor_path}")

        return model, processor

    def execute(self):
        image = self.load_image(self.image)

        # Preprocess the image using the processor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Run the model with no gradient computation -> inference modus
        with torch.no_grad():
            predictions = self.model(**inputs)

        # Get original image size in (height, width) format
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)

        # raw  output
        output = self.processor.post_process_object_detection(predictions, target_sizes=target_sizes)[0]

        results = []

        # Iterate over all detected boxes
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            # Keep only predictions with score >= 0.5
            if score >= 0.5:
                # Round box values for readability
                box = [round(float(x), 2) for x in box]
                label = int(label)
                # save result with box, label name and score
                results.append({
                    "box": box,
                    "label": label,
                    "label_name": self.label_map.get(label, f"Class {label}"),
                    "score": round(float(score), 4)
                })

        if self.log:
            print("### Results saved to layout_results.json")
            print(f"###  Detected layout elements: {len(results)}")

        image_with_boxes = draw_boxes_on_image(image, results)
        return image_with_boxes, results

    def load_image(self, image_input):
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError("Invalid image input")


def draw_boxes_on_image(image, detections, color=(0, 0, 255), thickness=2):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    if isinstance(image, np.ndarray) and image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    output = image.copy()

    for det in detections:
        box = det["box"]
        label_name = det.get("label_name", f"Class {det['label']}")
        score = det["score"]

        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        text = f"{label_name}: {score:.2f}"
        cv.putText(output, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return output
