from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import cv2 as cv
import numpy as np
from skimage.filters import threshold_sauvola
from src.util import get_ocr_score


class StepBinarize(AbstractPreprocessPipelineStep):
    # Binarization,tri different methods and select the one with the best OCR Result (tested with tesseract in util-function).
    def apply(self):
        # Slight blur to reduce small noise
        gray = cv.GaussianBlur(self.image, (3, 3), 0)

        results = {}

        # Try all methods
        self.try_otsu(gray, results)
        self.try_sauvola(gray, results)
        self.try_hybrid(results)
        self.try_sauvola_plus(results)

        # Select method with the best OCR score
        if results:
            best_method = None
            best_score = -1
            best_image = gray
            for method_name, (score, img) in results.items():
                if score > best_score:
                    best_score = score
                    best_method = method_name
                    best_image = img
        else:
            best_method = None
            best_score = 0
            best_image = gray

        if self.log and best_method:
            print(f"### Using {best_method} (OCR score: {best_score})")

        return best_image

    def try_sauvola(self, image, results):
        # Try Sauvola binarization
        try:
            window_size = 35  # Size of the region used to calculate local threshold
            thresh = threshold_sauvola(image, window_size=window_size)  # Apply threshold: if pixel > local threshold -> white (255), else black (0)
            binary = (image > thresh).astype(np.uint8) * 255

            score = get_ocr_score(binary)

            if self.log:
                print(f"###  Sauvola - OCR score: {score}")
            results["Sauvola"] = (score, binary)
        except Exception as e:
            if self.log:
                print(f"### [Error] in Sauvola: {e}")

    def try_otsu(self, image, results):
        # Try Otsu binarization.
        try:
            # OpenCV finds best threshold that separates foreground and background
            _, binary = cv.threshold(
                image,  # Input grayscale image
                0,  # Initial threshold
                255,  # Max value (white)
                cv.THRESH_BINARY + cv.THRESH_OTSU  # Use binary + Otsu method
            )
            score = get_ocr_score(binary)

            if self.log:
                print(f"### Otsu - OCR score: {score}")
            results["Otsu"] = (score, binary)
        except Exception as e:
            if self.log:
                print(f"### [Error] in Otsu: {e}")

    def try_hybrid(self, results):
        # Try combining Sauvola and Otsu with OR.
        try:
            if "Sauvola" in results and "Otsu" in results:
                sauvola_img = results["Sauvola"][1]  # get from previous try
                otsu_img = results["Otsu"][1]  # get from previous try
                binary = cv.bitwise_or(sauvola_img,
                                       otsu_img)  # Combine both results: highlight text areas detected by either method

                score = get_ocr_score(binary)

                if self.log:
                    print(f"### Hybrid - OCR score: {score}")
                results["Hybrid"] = (score, binary)
        except Exception as e:
            if self.log:
                print(f"### [Error] in Hybrid: {e}")

    def try_sauvola_plus(self, results):
        # Try line enhancement on Sauvola result.
        try:
            if "Sauvola" in results:
                binary = results["Sauvola"][1]

                # rectangular kernels for detecting horizontal and vertical lines
                h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
                v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))

                # Detect horizontal and vertical lines, morphological opening to isolate long lines
                h_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, h_kernel)
                v_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, v_kernel)

                # Add line structures to image
                lines = cv.bitwise_or(h_lines, v_lines)
                boosted = cv.bitwise_or(binary, lines)

                score = get_ocr_score(boosted)

                if self.log:
                    print(f"### Sauvola-Plus - OCR score: {score}")
                results["Sauvola-Plus"] = (score, boosted)
        except Exception as e:
            if self.log:
                print(f"### [Error] in Sauvola-Plus: {e}")
