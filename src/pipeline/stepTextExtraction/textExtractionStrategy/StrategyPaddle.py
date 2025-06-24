from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np

from src.pipeline.stepTextExtraction.textExtractionStrategy.AbstractStrategyTextExtraction import AbstractStrategyTextExtraction


class StrategyPaddle(AbstractStrategyTextExtraction):
    def __init__(self, image, log: bool = False):
        super().__init__(image, log)

    def execute(self):
        # Create copy of image for drawing bounding boxes
        image_copy = self.image.copy()
        if len(image_copy.shape) == 2:  # convert grayscale to BGR
            image_copy = cv.cvtColor(image_copy, cv.COLOR_GRAY2BGR)

        # OCR directly on the image (not file path)
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='de')  # Initialize PaddleOCR: eith angle classification, German language
        results = ocr_engine.ocr(self.image, det=True, cls=True)

        if not results or not results[0]:
            if self.log:
                print("### PaddleOCR returned no results.")
            return self.image, []

        ocr_result = []

        for line_id, line in enumerate(results[0]):  # for each detected text block
            polygon, (text, score) = line
            polygon = np.array(polygon).astype(np.int32) #converts polygon in numpyArray

            # Convert to rect-bbox: [x_min, y_min, x_max, y_max]
            x_coords = [pt[0] for pt in polygon]
            y_coords = [pt[1] for pt in polygon]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            # Append structured data to result list
            ocr_result.append({
                "text": text,
                "bbox": bbox,
                "confidence": float(score),
            })

            cv.polylines(image_copy, [polygon], isClosed=True, color=(255, 0, 0), thickness=4)
            cv.putText(image_copy, text, tuple(polygon[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 100), 2)
        cv.putText(image_copy, "Paddle", (200, 200), cv.FONT_HERSHEY_SIMPLEX, 3.2, (0, 0, 255), 5, cv.LINE_AA)

        # Return annotated image, structured OCR results
        return image_copy, ocr_result, None
