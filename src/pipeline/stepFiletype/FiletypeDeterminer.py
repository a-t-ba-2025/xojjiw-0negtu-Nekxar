import os
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from dotenv import load_dotenv
from PIL import Image
import cv2 as cv
import joblib

load_dotenv()
input_folder = os.getenv('INPUT_PATH')
TARGET_IMAGE_SIZE = (2480*2, 3508*2)  # A3 with 300 DPI


class FiletypeDeterminer:
    def __init__(self, upload_file, log: bool = False):
        self.upload_file = upload_file
        self.log = log

    def __enter__(self):
        print(f"# [Pipeline] [{self.__class__.__name__}] started: {self.upload_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"# [Pipeline] [{self.__class__.__name__}] completed: {self.upload_file}")
            print('- - -')

    def run(self):
        # get absolute file path and extension
        file_path = os.path.join(input_folder, self.upload_file)
        name, ext = os.path.splitext(self.upload_file.lower())

        ext = ext.lower()
        if ext == ".pdf":
            return self.process_pdf(name, file_path)
        if ext in [".jpg", ".jpeg", ".png"]:
            return self.process_image(name, file_path)
        else:
            raise ValueError(f"Unknown file type: {ext}")

    def process_pdf(self, name, file_path):
        # check if PDF contains real (machine-readable) text
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    print(f"## [Pipeline] [{self.__class__.__name__}] File is real (text-based) PDF")
                    images = convert_from_path(file_path, dpi=300)
                    image = np.array(images[0])  # Erste Seite als Bild
                    return name, "pdf", [file_path], image, None

        # otherwise: scanned PDF → convert to PNG images
        print(f"## [Pipeline] [{self.__class__.__name__}] File is scanned PDF -> converting to PNG and process as image")

        images = convert_from_path(file_path, dpi=300)
        if not images:
            raise RuntimeError("No images extracted from PDF.")

        # Save the first page as PNG to same input folder
        out_path = os.path.join(input_folder, f"{name}.png")
        images[0].save(out_path, "PNG")

        result = self.process_image(name, out_path)

        if os.path.exists(out_path):
            os.remove(out_path)

        return result

    def process_image(self, name, file_path):
        # convert image to RGB and save it as PNG with 300 DPI
        image = Image.open(file_path).convert("RGB")
        output_path = os.path.join(input_folder, f"{name}.png")
        image.save(output_path, dpi=(300, 300))

        # Convert the image from PIL to NumPy array for OpenCV compatibility
        image = np.array(image)
        image = cv.resize(image, TARGET_IMAGE_SIZE, interpolation=cv.INTER_AREA)
        is_mostly_text = self.is_mostly_text(image)
        print(f"### [Pipeline] [{self.__class__.__name__}] is mostly text : {is_mostly_text} -> {self.upload_file} ")
        return name, "image", [output_path], image, is_mostly_text

    def is_mostly_text(self, np_image):
        # Load the trained classifier model from environment path
        model_path = os.path.join(
            os.getenv('TEXT_MODEL'),
            "likely_text_model.pkl"
        )
        model = joblib.load(model_path)
        # Extract features from image
        features = self.extract_features(np_image)
        # Predict if image is mostly text (1 -> yes, 0 -> no)
        prediction = model.predict([features])[0]
        return prediction == 1

    def extract_features(self, np_image):
        gray = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY) # Convert image to grayscale
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        inverted = 255 - binary

        # Sum white pixels horizontal (text line projection)
        projection = np.sum(inverted, axis=1)
        # Normalize projection to 0–1
        norm_proj = projection / np.max(projection)
        # Count lines with many white pixles
        active_lines = np.sum(norm_proj > 0.2)

        # Find indices of active lines and get gaps
        indexes = np.where(norm_proj > 0.2)[0]
        gaps = np.diff(indexes)

        # Get standard deviation of line spacing
        std_gap = np.std(gaps) if len(gaps) > 0 else 0
        # Get ratio of white pixels in  image ->text density
        ratio_black = np.sum(inverted > 0) / inverted.size

        return [active_lines, std_gap, ratio_black]
