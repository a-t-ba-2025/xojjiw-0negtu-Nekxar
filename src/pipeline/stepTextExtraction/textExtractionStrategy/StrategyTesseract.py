from src.pipeline.stepTextExtraction.textExtractionStrategy.AbstractStrategyTextExtraction import AbstractStrategyTextExtraction
import pytesseract
from dotenv import load_dotenv
from pytesseract import Output
import cv2 as cv
import os

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")


class StrategyTesseract(AbstractStrategyTextExtraction):
    def __init__(self, image, log: bool = False):
        super().__init__(image, log)

    def execute(self):
        # Make copy of image for visualization
        image_copy = self.image.copy()
        # needs to be 3-channel BGR for drawing
        if len(image_copy.shape) == 2:  # if grayscale
            image_copy = cv.cvtColor(image_copy, cv.COLOR_GRAY2BGR)

        lang = 'deu'
        custom_config = r'--oem 1 --psm 3' #tesseract configuration: --oem 1: LSTM OCR engine, --psm 3: Fully automatic page segmentation, but no OSD. (Default)
        # Run tesseractOCR
        data = pytesseract.image_to_data(self.image, lang=lang, config=custom_config, output_type=Output.DICT)

        ocr_result = []
        num_boxes = len(data['text']) # Number of text boxes

        for i in range(num_boxes): # Loop each box
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            if text and conf > 0:  # filter empty or invalid results
                # bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [x, y, x + w, y + h]  # calculated coorinades

                # collect text, bounding box, confidence
                ocr_result.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": conf
                })

                cv.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 4) # Draw bounding box
                # draw the recognized text
                cv.putText(image_copy, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 100), 2)
        cv.putText(image_copy, "Tesseract", (200, 200), cv.FONT_HERSHEY_SIMPLEX, 3.2, (0, 0, 255), 5, cv.LINE_AA)

        return image_copy, ocr_result, None
